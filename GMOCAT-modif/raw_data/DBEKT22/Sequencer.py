import pandas as pd
import os
import json
import re
import numpy as np
import argparse

def generate_sequences(input_data, sequence_length, output_dir, delimeter=",", use_padding=True, padding_char="-1"):
    """Generates student practice sequences in a JSON format from Questions.csv and Transaction.csv data files given user's arguments.

    Args:
        input_data (str): Path for the input directory containing .csv files of the dataset.
        sequence_length (int): The user prefered max sequence length to use while generating sequences.
        output_dir (str): Path for the output directoy to use when saving the result sequences JSON file.
        delimeter (str): A delimeter character to use for separating items in the sequence.
        use_padding (bool): Whether or not to pad sequences with lengths shorter than the sequence_length argument.
        padding_char (str): The character to use for padding short sequences if padding is activated.

    Output:
        A JSON file containing student practice sequences in the dataset based on the user's arguments.

    """
    input_df = input_data.copy()
    input_df['start_time'] = pd.to_datetime(
        input_df['start_time'], infer_datetime_format=True, errors='coerce')
    input_df['end_time'] = pd.to_datetime(
        input_df['end_time'], infer_datetime_format=True, errors='coerce')
    input_df['time_taken'] = (input_df['end_time'] -
                              input_df['start_time']).dt.total_seconds()
    
    input_df.sort_values(by=['student_id', 'start_time'], inplace=True)
    json_record = {"student_id": 0, "seq_len": 0,
                   "question_ids": "", "answers": ""}
    sequences = []
    grouped_df = input_df.groupby('student_id')
    for group_name, group_df in grouped_df:
        seq_counter = 0
        question_seq = []
        answer_seq = []
        time_taken = []
        gt_difficulty = []
        difficulty_feedback = []
        answer_conf = []
        hint_used = []
        ans_changes = []

        for idx, row in group_df.iterrows():
            if seq_counter == sequence_length:
                sequences.append({"student_id": group_name,
                                  "seq_len": len(question_seq),
                                  "question_ids": delimeter.join(question_seq),
                                  "answers": delimeter.join(answer_seq),
                                  "gt_difficulty": delimeter.join(gt_difficulty),
                                  "difficulty_feedback": delimeter.join(difficulty_feedback),
                                  "answer_confidence": delimeter.join(answer_conf),
                                  "hint_used": delimeter.join(hint_used),
                                  "time_taken": delimeter.join(time_taken),
                                  "num_ans_changes": delimeter.join(ans_changes)})
                seq_counter = 0
                question_seq = []
                answer_seq = []
                time_taken = []
                gt_difficulty = []
                difficulty_feedback = []
                answer_conf = []
                hint_used = []
                ans_changes = []
            else:
                question_seq.append(str(row['question_id']))
                answer_seq.append("1" if row['answer_state'] else "0")
                time_taken.append(str(0.0 if row['time_taken'] <= 0 else np.round(row['time_taken'])))
                gt_difficulty.append(str(row['difficulty']))
                difficulty_feedback.append(str(row['difficulty_feedback']))
                answer_conf.append(str(row['trust_feedback']))
                hint_used.append(str(row['hint_used']))
                ans_changes.append(str(row['selection_change']))
                seq_counter += 1

        # Append any residuals
        residula_len = len(question_seq)
        if residula_len > 0:

            if use_padding:
                padd_seq=[padding_char for _ in range(sequence_length-residula_len)]
                question_seq.extend(padd_seq)
                answer_seq.extend(padd_seq)
                time_taken.extend(padd_seq)
                gt_difficulty.extend(padd_seq)
                difficulty_feedback.extend(padd_seq)
                answer_conf.extend(padd_seq)
                hint_used.extend(padd_seq)
                ans_changes.extend(padd_seq)

            sequences.append({"student_id": group_name,
                              "seq_len": residula_len,
                              "question_ids": delimeter.join(question_seq),
                              "answers": delimeter.join(answer_seq),
                              "gt_difficulty": delimeter.join(gt_difficulty),
                              "difficulty_feedback": delimeter.join(difficulty_feedback),
                              "answer_confidence": delimeter.join(answer_conf),
                              "hint_used": delimeter.join(hint_used),
                              "time_taken": delimeter.join(time_taken),
                              "num_ans_changes": delimeter.join(ans_changes)})

    with open(os.path.join(output_dir, "practice_sequences.json"), 'w+') as wf:
        json.dump(sequences, wf, indent=6)


if __name__ == "__main__":

    # Add runtime arguments
    parser = argparse.ArgumentParser(
        description='Commandline arguments for the sequencer script')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path for the data files directory',
                        required=True)
    parser.add_argument('--sequence_length',
                        type=int,
                        help='Max length for exercise sequence in the result',
                        required=True)
    parser.add_argument('--use_padding',
                        type=bool,
                        help='Whether or not to use padding for sequences shorter than the sequence_length argument',
                        required=True)
    parser.add_argument('--padding_char',
                        type=str,
                        help='The character to use for sequence padding if enabled',
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Path for the output directory',
                        required=True)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir), "incorrect data directory"
    assert os.path.exists(args.output_dir), "incorrect output directory"

    # Load question and transactions dataframes from the input dir
    questions_df = pd.read_csv(os.path.join(
        args.data_dir, "Questions.csv"), encoding='utf-8')
    transactions_df = pd.read_csv(os.path.join(
        args.data_dir, "Transaction.csv"), encoding='utf-8')
    data_df = pd.merge(questions_df, transactions_df,
                       how='inner', left_on='id', right_on='question_id')

    # Call generate sequences
    generate_sequences(input_data=data_df,
                       sequence_length=args.sequence_length,
                       output_dir=args.output_dir,
                       use_padding=args.use_padding,
                       padding_char=args.padding_char)

    print("Exercise sequences file has been successfully generated and saved into the output dir.")
