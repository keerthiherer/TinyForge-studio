import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copy2
import random
import mimetypes

def split_index(file_count, test_size):
	if file_count <= 1:
		return file_count
	return max(1, min(file_count - 1, int(file_count * (1 - test_size))))

def split_csv(file_path, output_dir, target_column='target', test_size=0.2, random_state=42):
	os.makedirs(output_dir, exist_ok=True)
	data = pd.read_csv(file_path)
	if target_column not in data.columns:
		raise ValueError(f"Target column '{target_column}' was not found in {file_path}.")
	X = data.drop(target_column, axis=1)
	y = data[target_column]
	stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=stratify
	)
	X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
	X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
	y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
	y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
	print('CSV data split and stored.')

def split_files(input_dir, output_dir, test_size=0.2, random_state=42):
	train_dir = os.path.join(output_dir, 'train')
	test_dir = os.path.join(output_dir, 'test')
	os.makedirs(train_dir, exist_ok=True)
	os.makedirs(test_dir, exist_ok=True)
	class_dirs = [
		d for d in os.listdir(input_dir)
		if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')
	]
	if class_dirs:
		for class_name in class_dirs:
			class_path = os.path.join(input_dir, class_name)
			files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
			if not files:
				continue
			random.seed(random_state)
			random.shuffle(files)
			split_idx = split_index(len(files), test_size)
			for target_root, selected_files in ((train_dir, files[:split_idx]), (test_dir, files[split_idx:])):
				target_class_dir = os.path.join(target_root, class_name)
				os.makedirs(target_class_dir, exist_ok=True)
				for f in selected_files:
					copy2(os.path.join(class_path, f), target_class_dir)
	else:
		files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
		if not files:
			raise ValueError(f"No files found in {input_dir}.")
		random.seed(random_state)
		random.shuffle(files)
		split_idx = split_index(len(files), test_size)
		train_files = files[:split_idx]
		test_files = files[split_idx:]
		for f in train_files:
			copy2(os.path.join(input_dir, f), train_dir)
		for f in test_files:
			copy2(os.path.join(input_dir, f), test_dir)
	print('Files split and stored in train/test folders.')

def detect_and_split(input_path, output_dir, target_column='target', test_size=0.2, random_state=42):
	if os.path.isdir(input_path):
		# Assume directory contains images, audio, or numeric files
		split_files(input_path, output_dir, test_size, random_state)
	elif os.path.isfile(input_path):
		mime, _ = mimetypes.guess_type(input_path)
		if input_path.lower().endswith('.csv') or (mime and 'csv' in mime):
			split_csv(input_path, output_dir, target_column, test_size, random_state)
		else:
			raise ValueError(f'Unsupported file type: {mime}. Please provide a CSV or a directory of files.')
	else:
		raise FileNotFoundError('Input path does not exist.')

if __name__ == '__main__':
	# Example usage:
	# For CSV: python split_and_store_data.py data.csv output_dir --target target_column
	# For files: python split_and_store_data.py data_folder output_dir
	import argparse
	parser = argparse.ArgumentParser(description='Split data for ML preprocessing.')
	parser.add_argument('input_path', help='Path to CSV file or directory of files (images, audio, etc.)')
	parser.add_argument('output_dir', help='Directory to store split data')
	parser.add_argument('--target', default='target', help='Target column name for CSV files')
	parser.add_argument('--test_size', type=float, default=0.2, help='Test set size fraction')
	parser.add_argument('--random_state', type=int, default=42, help='Random seed')
	args = parser.parse_args()
	detect_and_split(args.input_path, args.output_dir, args.target, args.test_size, args.random_state)
