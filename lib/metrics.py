import pandas as pd
from rich import print
import csv


class Model_metrics:
    def __init__(self, data_path):
        """
        Initialize the class with the path of the csv file.

        Parameters:
            -data_path (str) : Path of the csv file.
        Returns:
            None
        """
        self.path = data_path
        self.index = 0
        self.met_dict = dict()
        self.dictionary = {
            "bertscore": 0,
            "bertscore_precision": 0,
            "bertscore_recall": 0,
            "bleu": 0,
            "rouge": 0,
            "meteor": 0,
        }

    def path_name(self):
        """
        Return the path of the csv file
        """
        return self.path

    def adjust_csv(self, num_lines):
        """
        Adjust the csv file by removing the last num_lines lines.

        Parameters:
            -num_lines (int) : Number of lines to remove from the csv file.

        Returns:
            None
        """
        dataframe = pd.read_csv(self.path, header=None)
        df_mod = dataframe.iloc[:-num_lines]
        df_mod.to_csv(self.path, index=False, header=False)

    def get_metrics(self):
        """
        Return the metrics in a dictionary format.
        :return: Dictionary of metrics.
        """
        for key, value in self.dictionary.items():
            self.met_dict[key] = value / self.index
        return self.met_dict

    def get_len(self):
        """
        Return the number of elements in the csv file
        :return: int
        """
        return self.index

    def get_dictionary(self):
        """
        Return the dictionary with the sum of the metrics
        :return: dictionary
        """
        return self.dictionary

    def read(self):
        """
        Read the csv file and store the metrics in a dictionary
        :return: None
        """
        with open(self.path, newline="") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                self.dictionary["bertscore"] += float(row[0])
                self.dictionary["bertscore_precision"] += float(row[1])
                self.dictionary["bertscore_recall"] += float(row[2])
                self.dictionary["bleu"] += float(row[3])
                self.dictionary["rouge"] += float(row[4])
                self.dictionary["meteor"] += float(row[5])
                self.index += 1

    def save_metrics(self, name_file: str):
        """
        Save the metrics in a csv file
        :Parameters: name_file
        :return: None
        """
        with open(name_file, "w") as f:
            w = csv.writer(f)
            w.writerow(self.met_dict.keys())
            w.writerow(self.met_dict.values())


if __name__ == "__main__":
    llama = Model_metrics("llama2.csv")
    # llama.adjust_csv(1870)
    llama.read()
    print(llama.get_metrics())
    print(llama.get_len())
    llama.save_metrics("llama2_metrics.csv")
