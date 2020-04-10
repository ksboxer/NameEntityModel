import yaml
from classes.TrainingParser import *
from classes.FeatureBuilder import *
import numpy as np
#import nltk
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify import MaxentClassifier

from scipy.sparse import csr_matrix

from sklearn import metrics

def run():
  with open('configs.yaml') as f:
      configs = yaml.load(f)
  #print(configs)

  # read in configs
  training_data_filepath = configs['training_data_file']
  development_data_filepath_x = configs['development_data_file_x']
  development_data_filepath_y = configs['development_data_file_y']
  test_file = configs['testfile']

  # city and names
  city_path = configs['world_cites']
  names_path = configs['names']


  # parse training data
  parser = Parser()
  raw_training_data = parser.parse_training_data(training_data_filepath)
  raw_developement_data = parser.parse_development_data(development_data_filepath_x, development_data_filepath_y)
  raw_testing_data = parser.parse_training_data(test_file)
  print(raw_testing_data[0:10])


  features_train = FeatureBuilder(raw_training_data, city_path, names_path)
  data_preprocessed_train = features_train.build_training_data(labels_v = True)
  #print(data_preprocessed_train[0:10])

  features_dev = FeatureBuilder(raw_developement_data, city_path, names_path)
  data_preprocessed_dev = features_dev.build_training_data(labels_v = True)
  #print(data_preprocessed_dev[0:10])

  features_testing = FeatureBuilder(raw_testing_data,  city_path, names_path, labels = False)
  data_processed_testing = features_testing.build_training_data(labels_v = False)
  #print(data_processed_testing[0:10])


  data_preprocessed_train_formatted = [(features_train.build_features_two(element[0]), element[1]) for element in data_preprocessed_train]
  data_preprocessed_dev_formatted = [(features_train.build_features_two(element[0]), element[1]) for element in data_preprocessed_dev]
  data_preprocessed_testing_formatted = [(features_train.build_features_two(element[0]), element[1]) for element in data_processed_testing]

  alg = MaxentClassifier.ALGORITHMS[0]
  iters = 15
  MaxEnt_classifier = MaxentClassifier.train(data_preprocessed_train_formatted, alg, max_iter=iters)

  actual_labels = []
  predicted_labels = []
  for unit in data_preprocessed_dev_formatted:
      label = unit[1]
      doc = unit[0]
      predicted_label = MaxEnt_classifier.classify(doc)
      actual_labels.append(label)
      predicted_labels.append(predicted_label)
      #print('predicted: '+ predicted_label + "  actual label: "+ label )

  print(metrics.confusion_matrix(actual_labels, predicted_labels))
  print(metrics.classification_report(actual_labels,predicted_labels))


  # training with training and developmental data_preprocessed
  full_data = data_preprocessed_train_formatted + data_preprocessed_dev_formatted

  MaxEnt_classifier = MaxentClassifier.train(full_data, alg, max_iter=iters)

  predicted_labels = []
  for unit in data_preprocessed_testing_formatted:
      doc = unit[0]
      predicted_label = MaxEnt_classifier.classify(doc)
      predicted_labels.append(predicted_label)

  rows = []
  idx = 0
  for sentence in raw_testing_data:
      if sentence[0] == 'end-of-sentence-here-07039':
          rows.append("")
      else:
          sentence.append(predicted_labels[idx])
          rows.append("\t".join(sentence))
      idx = idx + 1

  with open('output_initial_feature_set_punct.txt', 'w') as f:
      for item in rows:
          f.write("%s\n" % item)



run()
