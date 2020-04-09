import yaml
from classes.TrainingParser import *
from classes.FeatureBuilder import *
import numpy as np
#import nltk
from sklearn.linear_model import LogisticRegression, SGDClassifier
#from nltk.classify import MaxentClassifier

from scipy.sparse import csr_matrix

def run():
  with open('configs.yaml') as f:
      configs = yaml.load(f)
  print(configs)

  # read in configs
  training_data_filepath = configs['training_data_file']
  development_data_filepath_x = configs['development_data_file_x']
  development_data_filepath_y = configs['development_data_file_y']


  # parse training data
  parser = Parser()
  raw_training_data = parser.parse_training_data(training_data_filepath)

  raw_developement_data = parser.parse_development_data(development_data_filepath_x, development_data_filepath_y)


  features_train = FeatureBuilder(raw_training_data)
  data_preprocessed_train = features_train.build_training_data(labels_v = True)

  features_dev = FeatureBuilder(raw_developement_data)
  data_preprocessed_dev = features_dev.build_training_data(labels_v = True)

  features_dev.map_training_to_dev(data_preprocessed_train, data_preprocessed_dev)



  #columnnames = list(data_preprocessed)
  #columnnames.remove('labels')
  #x = data_preprocessed[columnnames]
  #y = data_preprocessed['labels']

  #print(len(x))

  #print('training classifier')
  #clf = SGDClassifier(loss = 'log', max_iter = 5, n_jobs = -1)
  #clf.partial_fit(x,y, classes=np.unique(y))


  ###### working on development data_preprocessed


  #trainset = []

  #idx = 0
  #for row in x:
#      trainset.append((row,y[idx]))
#      idx = idx + 1


 # print('done preparing training set')
 # max_ent_clf = MaxentClassifier(trainset)

  # building hmm model

  #hmm = HMM()
  #hmm.build_hmm(raw_training_data)
  #hmm_data, known_words, total_words = hmm.get_hmm_data()

  # running viterbi on data
  #viterbi_runner = Viterbi(hmm_data, known_words, 'uniform', total_words)
  #viterbi_runner.run_viterbi(["He","looked","through", "his","pile","of","marxists","books", '.'])

  #raw_testing_data = parser.parse_test_data(testing_data_filepath)
  #print(raw_testing_data)
  #output = []
  #for sentence in raw_testing_data:
    #  print(sentence)
     # words, pos = viterbi_runner.run_viterbi(sentence)
     # for i in range(0, len(words)):
    #      output.append(words[i] + "\t" + pos[i])
    #      print(words[i] + "\t" + pos[i])
    #  output.append("")
#
 # with open('wsj_23.pos', 'w') as filehandle:
#      filehandle.writelines("%s\n" % place for place in output)
 # print(output)

run()
