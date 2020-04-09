
import pandas as pd

class FeatureBuilder:

  def __init__(self, raw_training_data):
    self.raw_training_data = raw_training_data
    self.reconstruct_sentences()
    #print('Done reconstruction sentences')
    #print(len(self.sentences))
    #print(len(self.pos_accum))
    #print(len(self.chunk_accum))
    #print(len(self.labels_accum))
    #print(self.sentences[1])


  def reconstruct_sentences(self):
      # sentences
      self.sentences  = []
      self.pos_accum = []
      self.chunk_accum = []
      self.labels_accum = []
      sentence = []
      pos = []
      chunk = []
      label = []
      for row in self.raw_training_data:
          if row[0] == "end-of-sentence-here-07039":
              self.sentences.append(sentence)
              sentence = []
              self.pos_accum.append(pos)
              pos = []
              self.chunk_accum.append(chunk)
              chunk = []
              self.labels_accum.append(label)
              label = []
          else:
              #print(row)
              sentence.append(row[0])
              pos.append(row[1])
              chunk.append(row[2])
              label.append(row[3])

  def build_training_data(self, labels_v = True):
      training_set = []

      if labels_v == True:
          labels = []

      s_idx = 0
      for sentence in self.sentences:
          idx = 0
          while idx < len(sentence):
              doc = []
              if idx == 0 :
                  doc.append("none-first-word-placeholder")
                  doc.append('none-first-word-placeholder-pos')
                  doc.append('none-first-word-placeholder-chunk')
              else:
                  doc.append(sentence[idx -1])
                  doc.append(self.pos_accum[s_idx][idx-1])
                  doc.append(self.chunk_accum[s_idx][idx-1])

              doc.append(sentence[idx])
              doc.append(self.pos_accum[s_idx][idx])
              doc.append(self.chunk_accum[s_idx][idx])

              if labels_v == True:
                  label = self.labels_accum[s_idx][idx]

              if idx + 1 >= len(sentence):
                  doc.append("none-last-word-placeholder")
                  doc.append('none-last-word-pos-placeholder')
                  doc.append('none-last-word-chunk-placeholder')
              else:
                  doc.append(sentence[idx+1])
                  doc.append(self.pos_accum[s_idx][idx+1])
                  doc.append(self.chunk_accum[s_idx][idx+1])
              training_set.append((doc, label))
              idx = idx + 1
          s_idx = s_idx + 1
          if s_idx % 10000 == 0:
              print(str(s_idx)+ " out of "+ str(len(self.sentences)))

      return(training_set)

  def build_features_two(self,doc):
      featureset = {}
      featureset['first_word(%s)' % doc[0]] = True
      featureset['first_word_pos(%s)' % doc[1]] = True
      featureset['first_word_chunk(%s)' % doc[2]] = True

      featureset['second_word(%s)' % doc[3]] = True
      featureset['second_word_pos(%s)' % doc[4]] = True
      featureset['second_word_chunk(%s)' % doc[5]] = True

      featureset['third_word(%s)' % doc[6]] = True
      featureset['third_word_pos(%s)' % doc[7]] = True
      featureset['third_word_chunk(%s)' % doc[8]] = True

      return(featureset)

  def map_training_to_dev(self, training_data, developement_data):
      training_columns = list(training_data)
      dev_columns = list(developement_data)

      cols_to_drop = []
      for dev_col in dev_columns:
          if dev_col not in training_columns:
              cols_to_drop.append(dev_col)
      print(len(cols_to_drop))
      print(len(developement_data.columns))
      developement_data = developement_data.drop(cols_to_drop, axis = 1)
      print(len(developement_data.columns))

      columns_to_append = []

      for training_col in training_columns:
          if training_col not in dev_columns:
              columns_to_append.append(training_col)

      print(len(columns_to_append))

      df = pd.DataFrame(0, index=range(len(developement_data)), columns=columns_to_append)

      developement_data = pd.concat([developement_data, df], axis = 1)



      developement_data = developement_data[training_columns]



      print(list(training_data) == list(developement_data))
      return(developement_data)
