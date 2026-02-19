import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class NCD(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.prednet_input_len = self.knowledge_dim

        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable

        super(NCD, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        # input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = e_discrimination * (stu_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))    
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def init_stu_emb(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'student' in name:
                nn.init.xavier_normal_(param)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class NCDModel:
    def __init__(self, args, num_students, num_questions, num_knowledges):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device) # Use device from args
        self.num_knowledges = num_knowledges
        self.model = NCD(num_students, num_questions, num_knowledges).to(self.device)
        self.loss_function = nn.BCELoss()

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_stu_emb(self):
        self.model.init_stu_emb()

    def cal_loss(self, sids, query_rates, concept_map):
        device = self.device
        real = []
        pred = []
        all_loss = np.zeros(len(sids))
        with torch.no_grad():
            self.model.eval()
            for idx, sid in enumerate(sids):
                question_ids = list(query_rates[sid].keys())
                student_ids = [sid] * len(question_ids)
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * self.num_knowledges
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)

                labels = [query_rates[sid][qid] for qid in question_ids]
                real.append(np.array(labels))

                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                labels = torch.LongTensor(labels).to(device)
                output = self.model(student_ids, question_ids, concepts_embs)
                loss = self._loss_function(output, labels).cpu().detach().numpy()
                all_loss[idx] = loss.item()
                pred.append(np.array(output.view(-1).tolist()))
            self.model.train()
        
        return all_loss, pred, real

    def get_knowledge_status(self, stu_ids, concept_map=None):
        """
        Returns knowledge status. If MC Dropout uncertainty is implemented,
        it returns both mean knowledge and uncertainty.
        For now, let's keep the return signature compatible or handle it in GCATEnv.
        """
        # If concept_map is provided, we can estimate uncertainty per concept
        if concept_map is not None:
             return self.estimate_concept_uncertainty(stu_ids, concept_map)

        stat_emb = torch.sigmoid(self.model.student_emb(stu_ids))
        return stat_emb.data

    def estimate_concept_uncertainty(self, stu_ids, concept_map, n_samples=5):
        """
        Estimate uncertainty of mastery for each concept for the given students using MC Dropout.
        We simulate 'probing' questions for each concept to measure predictive variance.
        """
        device = self.device
        self.model.train() # Enable dropout

        # We need to construct a set of virtual questions that cover all concepts.
        # Ideally, we want one question per concept.
        # But questions are embeddings.
        # We can construct a dummy batch where for each concept k, we create an input
        # that maximizes the sensitivity to k.
        # In NCD, input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb (if Q matrix)
        # Actually our NCD implementation takes concepts_embs as input!
        # forward(stu_id, exer_id, kn_emb)
        # We can pass dummy exer_id (e.g. 0) but control kn_emb.

        num_concepts = self.num_knowledges
        batch_size = len(stu_ids)

        # We want output (batch_size, num_concepts) containing uncertainty.
        uncertainties = torch.zeros(batch_size, num_concepts).to(device)
        means = torch.zeros(batch_size, num_concepts).to(device)

        # Since we can't easily batch all concepts for all users in one go due to memory,
        # let's iterate or process efficiently.
        # We can process all users against ONE concept at a time? No, too slow.
        # We can process all users against ALL concepts if we assume "questions" are just concepts.

        # Construct "Probing" inputs
        # For each concept k, we want to know variance of prediction if question had ONLY concept k.
        # We can simulate this by passing kn_emb with 1 at k and 0 elsewhere.
        # But we also need an exer_id. The difficulty/discrimination matters.
        # If we pick a random exer_id, it biases the uncertainty.
        # We can create a "neutral" exercise embedding? No, we have to pick an index.
        # Let's pick a frequent exercise for each concept or just use padding index if valid?
        # Or better: Average the difficulty/discrimination of all questions having that concept?
        # That's hard with current architecture (Embeddings).

        # Approximation: Pick the first question available in concept_map that covers concept k.
        # If no question covers k only, pick one that covers k.

        concept_to_qid = {}
        # concept_map maps qid -> list of concepts
        # We need concept -> qid
        # concept_map keys are strings or ints? In GCATEnv it is loaded from json, keys are strings usually but casted.
        # Env.py: know_map[qid_pad] = concept_data[str(qid)]

        # Inverse map
        for qid, concepts in concept_map.items():
            if qid == 0: continue # Pad
            for c in concepts:
                if c not in concept_to_qid:
                    concept_to_qid[c] = qid

        # Prepare inputs
        # We will run M forward passes.
        # In each pass, we evaluate all (student, concept) pairs.
        # This effectively means we treat concepts as questions.

        # To do this efficiently:
        # We can expand students to (batch_size * num_concepts).
        # But that's large.
        # Let's do it per concept or batches of concepts?
        # Actually, let's just do it for the target concepts if possible?
        # The function signature is generic.

        # Let's simply iterate n_samples.
        # Inside, we need to get P(correct | s, c).
        # We use the 'concept_to_qid' to pick a representative question for 'c'.
        # Note: This measures uncertainty of the prediction on that question, which depends on concept mastery.

        # Better approach for NCD specifically:
        # The uncertainty we care about is the variance of the student embedding itself?
        # No, the student embedding is static (optimized).
        # The uncertainty comes from the fact that we don't know the TRUE mastery.
        # But NCD is a point estimate model.
        # The MC Dropout gives "Epistemic Uncertainty" of the model.
        # If the model relies heavily on dropout-prone features to predict for concept k, uncertainty is high.

        predictions_sum = torch.zeros(batch_size, num_concepts).to(device)
        predictions_sq_sum = torch.zeros(batch_size, num_concepts).to(device)

        # Mapping from concept index to representative question index
        q_indices = []
        c_embs_list = []

        valid_concepts = []
        for c in range(num_concepts):
            if c in concept_to_qid:
                q = concept_to_qid[c]
                q_indices.append(q)

                # Construct one-hot concept emb
                c_emb = [0.] * num_concepts
                c_emb[c] = 1.0
                c_embs_list.append(c_emb)
                valid_concepts.append(c)
            else:
                # Concept not in any question?
                q_indices.append(1) # Dummy
                c_embs_list.append([0.] * num_concepts)

        q_tensor = torch.LongTensor(q_indices).to(device) # (num_concepts,)
        c_tensor = torch.Tensor(c_embs_list).to(device)   # (num_concepts, num_concepts)

        # Expand for batch
        # s_ids: (batch_size,) -> (batch_size, num_concepts)
        # q_tensor: (num_concepts,) -> (batch_size, num_concepts)

        s_expanded = stu_ids.unsqueeze(1).repeat(1, num_concepts).view(-1)
        q_expanded = q_tensor.unsqueeze(0).repeat(batch_size, 1).view(-1)
        c_expanded = c_tensor.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, num_concepts)

        with torch.no_grad():
            for _ in range(n_samples):
                # Forward pass
                # model.forward(stu_id, exer_id, kn_emb)
                # Output: (batch_size * num_concepts, 1)
                out = self.model(s_expanded, q_expanded, c_expanded).view(batch_size, num_concepts)

                predictions_sum += out
                predictions_sq_sum += out ** 2

        # Calculate variance: E[X^2] - (E[X])^2
        means = predictions_sum / n_samples
        variance = (predictions_sq_sum / n_samples) - (means ** 2)

        self.model.eval() # Restore eval mode

        # Return means (as proxy for mastery score) and variance (uncertainty)
        # But wait, original get_knowledge_status returns `stat_emb` which is just student embedding weights.
        # `means` here is probability of correctness on a probe question.
        # This is different!
        # The environment uses `prev_score` vs `curr_score`.
        # If we switch to using `means` (probability), it's fine, but we should be consistent.
        # Actually, `stat_emb` from `get_knowledge_status` (original) returns the raw mastery values [0,1].
        # `means` here is P(correct). P(correct) is monotonic with mastery.
        # However, for consistency with other parts of the code that might use knowledge status,
        # maybe we should return the `stat_emb` as the "mean" and calculate uncertainty of `stat_emb`?
        # But `stat_emb` is deterministic!

        # Let's stick to the plan:
        # "Uncertainty-Based Termination ... berbasis MC Dropout uncertainty per konsep"
        # We will use the Variance calculated above as the uncertainty metric.
        # For the "score" (to replace know_stat), we can continue to use the student embedding
        # OR use the `means` calculated here.
        # Using `means` is safer because it reflects the model's predictive belief.
        # But `stat_emb` is what NCD learns directly.

        # Let's return a tuple or object?
        # The Env expects a tensor.
        # I will modify Env to handle tuple return or pack it.
        # Let's return stack: (mean, uncertainty) -> (batch, num_concepts, 2)

        # Wait, the original code:
        # self.know_stat = self.model.get_knowledge_status(...)
        # prev_score, curr_score = ...

        # I will return two tensors.

        # But wait, `means` is P(correct) on a specific question.
        # `stat_emb` is the latent trait.
        # Ideally we want uncertainty of latent trait.
        # But we can't get it easily.
        # So using predictive uncertainty is the standard proxy in Deep Learning.

        # I'll return `stat_emb` as the "score" (to keep consistency with plotting latent mastery if needed)
        # and `variance` as the "uncertainty".

        stat_emb = torch.sigmoid(self.model.student_emb(stu_ids)).data
        return stat_emb, variance
        
    def train(self, train_data, lr, batch_size, epochs, path):
        device = self.device
        logger = logging.getLogger("Pretrain")
        logger.info('train on {}'.format(device))
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_loss = 1000000

        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
            loss /= len(train_loader)
            logger.info('Epoch [{}]: loss={:.5f}'.format(ep, loss))
    
            if loss < best_loss:
                best_loss = loss
                logger.info('Store model')
                self.adaptest_save(path)
                # if cnt % log_step == 0:
                #     logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
    
    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.device) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)
    
    def adaptest_save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)
    
    def adaptest_load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        self.model.to(self.device)
    
    def update(self, tested_dataset, lr, epochs, batch_size):
        device = self.device
        optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            # log_steps = 100
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))

    def get_pred(self, user_ids, avail_questions, concept_map):
        device = self.device

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in user_ids:
                pred_all[sid] = {}
                question_ids =  list(avail_questions[sid])
                student_ids = [sid] * len(question_ids)
          
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[str(qid)]
                    concepts_emb = [0.] * self.num_knowledges
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(avail_questions[sid])):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, pred_all: dict, concept_map):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        # epochs = self.args.cdm_epoch
        epochs = 1
        # lr = self.args.cdm_lr
        lr = 0.01
        device = self.device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = concept_map[str(qid)]
        concepts_emb = [0.] * self.num_knowledges
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        # pred = self.model(student_id, question_id, concepts_emb).item()
        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()