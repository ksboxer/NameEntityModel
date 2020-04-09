
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
          if row[0] == "-DOCSTART-":
              pass
          elif row[0] == "end-of-sentence-here-07039":
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
      first_words = []
      second_words = []
      thirds_words = []

      first_word_pos = []
      second_word_pos = []
      third_word_pos = []

      first_word_chunk = []
      second_word_chunk = []
      third_word_chunk = []

      if labels_v == True:
          labels = []

      s_idx = 0
      for sentence in self.sentences:
          idx = 0
          while idx < len(sentence):
              if idx == 0 :
                  first_words.append("none-first-word-placeholder")
                  first_word_pos.append('none-first-word-placeholder-pos')
                  first_word_chunk.append('none-first-word-placeholder-chunk')
              else:
                  first_words.append(sentence[idx -1])
                  first_word_pos.append(self.pos_accum[s_idx][idx-1])
                  first_word_chunk.append(self.chunk_accum[s_idx][idx-1])

              second_words.append(sentence[idx])
              second_word_pos.append(self.pos_accum[s_idx][idx])
              second_word_chunk.append(self.chunk_accum[s_idx][idx])

              if labels_v == True:
                  labels.append(self.labels_accum[s_idx][idx])

              if idx + 1 >= len(sentence):
                  thirds_words.append("none-last-word-placeholder")
                  third_word_pos.append('none-last-word-pos-placeholder')
                  third_word_chunk.append('none-last-word-chunk-placeholder')
              else:
                  thirds_words.append(sentence[idx+1])
                  third_word_pos.append(self.pos_accum[s_idx][idx+1])
                  third_word_chunk.append(self.chunk_accum[s_idx][idx+1])
              idx = idx + 1
          s_idx = s_idx + 1
          if s_idx % 10000 == 0:
              print(str(s_idx)+ " out of "+ str(len(self.sentences)))

      df_features = pd.DataFrame({"w_i_1":first_words,
                                    "w_i": second_words,
                                    "w_i_p_1": thirds_words,
                                    "w_i_pos":second_word_pos,
                                    "w_i_1_pos" : first_word_pos,
                                    "w_i_p_1_pos" : third_word_pos,
                                    "w_i_chunk": second_word_chunk,
                                    "w_i_1_chunk": first_word_chunk,
                                    "w_i_p_chunk": third_word_pos})
      if labels_v == True:
          df_features['labels'] = labels

      column_names = list(df_features)
      if labels_v == True:
          column_names.remove("labels")
      print(column_names)
      df_final = pd.DataFrame()

      if labels_v == True:
          f = [df_features['labels']]
      else:
          f = []

      for col in column_names:
          print(col)
          f.append(pd.get_dummies(df_features[col], sparse = True, prefix = col))

      temp = pd.concat(f, axis = 1)
      return temp


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
