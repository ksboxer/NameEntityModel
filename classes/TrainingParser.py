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
