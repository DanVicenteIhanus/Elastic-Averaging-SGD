import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

folder_path = "../data/"
plots_path = "../plots/"

# Original string format
original_string = "training_stats_size3_rank_0_tau_2_beta_3.96"

# Target string format
example_string = "workers = 2, $\\tau$ = 5, $\\beta$ = 3.98"

# Regular expression pattern to match numbers
pattern = r"(\d+(\.\d+)?)"
formatted_string = re.sub(pattern, "{}", example_string)

fontSize = 15

# Loop over each file in the directory
for filename in os.listdir(folder_path):
    # Check if the entry is a file
    if os.path.isfile(os.path.join(folder_path, filename)):
        #print(filename)  # Do whatever you want with the filename
        try:
            original_string = filename
            numbers = re.findall(pattern, original_string)
            del numbers[1]
            numbers[0] = (str(int(numbers[0][0])-1),'') #reduce workers by 1 (because of root)
            # Replace numbers with placeholders

            # Format the string with the extracted numbers
            new_string = formatted_string.format(*[number[0] for number in numbers])

            #print(new_string)
            df = pd.read_csv(folder_path + filename)
            df['Duration'] = df['Duration']/1000
            #Plot 1: accuracy
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(df['Duration'], df['Accuracy'], ":or", lw=0.5)
            plt.title('Training accuracy: ' + new_string)
            plt.xlabel('Wall-clock time (s)', fontsize=fontSize)
            plt.ylabel('Classification accuracy', fontsize=fontSize)
            plt.savefig(plots_path + original_string[:-3] + "_accuracy.pdf", bbox_inches="tight")
            plt.close()
            
            #Plot 2: loss
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(df['Duration'], df['Sample_Mean_Loss'], ":*b", lw=0.5)
            plt.title('Training loss: ' + new_string)
            plt.xlabel('Wall-clock time (s)', fontsize=fontSize)
            plt.ylabel('Mean loss', fontsize=fontSize)
            plt.savefig(plots_path + original_string[:-3] + "_loss.pdf", bbox_inches="tight")
            plt.close()
            pass
        except:
            pass
        