import yaml

class Parser:


  def parse_training_data(self, filename):
      all_data = []
      with open(filename) as f:
          lines = f.readlines()
      for line in lines:
          line = line.replace("\n", "")
          if line != "":
              split_elements = line.split("\t")
              all_data.append(split_elements)
          else:
              all_data.append(["end-of-sentence-here-07039"])
      return(all_data)

  def parse_test_data(self, filename):
    sentences = []
    with open(filename) as f:
        lines = f.readlines()

    sentence = []
    for line in lines:
        line = line.replace("\n", "")
        if line == "":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line)
    return sentences

  def parse_development_data(self, x_file, y_file):
    parsed_input = self.parse_training_data(x_file)
    parsed_output = self.parse_training_data(y_file)
    #print(parsed_input[0:20])
    #print(parsed_output[0:20])

    modified_sentences = []
    idx = 0
    for s in parsed_input:
        s.append(parsed_output[idx][-1])
        modified_sentences.append(s)
        idx = idx + 1
    #print(modified_sentences[0:20])
    return(modified_sentences)
