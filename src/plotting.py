import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

import shutil

def plotAllInFolder(folderPath,plotPath, isPlottingBenchmark,df_benchmark, hasMomentum,formattedString, formatPattern,isCifar, fontSize):
    dataset = "CIFAR" if isCifar else "MNIST"
    for filename in os.listdir(folderPath):
    # Check if the entry is a file
        if os.path.isfile(os.path.join(folderPath, filename)):
            original_string = filename
            isRoot = False
            numbers = re.findall(formatPattern, original_string)
            if len(numbers) >1:
                if numbers[1][0] == '0':
                    isRoot = True
                del numbers[1]
                if not hasMomentum:
                    del numbers[-1]
                numbers[0] = (str(int(numbers[0][0])-1),'') #reduce workers by 1 (because of root)

            # Format the string with the extracted numbers
            new_string = formattedString.format(*[number[0] for number in numbers])

            df = pd.read_csv(folderPath+ filename, sep = ",")
            df['Duration'] = df['Duration']/1000
            df = df.reset_index()
            df = df.rename(columns={"index": "Epoch"})
            df["Epoch"] = df["Epoch"] + 1
            
            # print(filename)
            # print(df.columns)
            # print(df.head())
            # print(df.iloc[:,2].head())
            #print(df['Accuracy'].head())
                
            #Plot 1: accuracy
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
        
            plt.plot(df['Epoch'], df.iloc[:,2], ":*b", lw=0.5,label='EAMSGD')
            if isPlottingBenchmark:
                plt.plot(df_benchmark['Epoch'], df_benchmark['Accuracy'], ":^r", lw=0.5,label='MSGD')
                plt.legend()
            plt.title(dataset + ' training accuracy: ' + new_string)
            plt.xlabel('Number of Epochs', fontsize=fontSize)
            plt.ylabel('Classification accuracy', fontsize=fontSize)
            plt.savefig(plotPath + original_string[:-3] + "_accuracy.pdf", bbox_inches="tight")
            plt.close()
            
            #Plot 2: loss
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(df['Epoch'], df.iloc[:,3], ":*b", lw=0.5,label='EAMSGD')
            if isPlottingBenchmark:
                plt.plot(df_benchmark['Epoch'], df_benchmark['Sample_Mean_Loss'], ":^r", lw=0.5,label='MSGD')
                plt.legend()
            plt.title(dataset + ' training loss: ' + new_string)
            plt.xlabel('Number of Epochs', fontsize=fontSize)
            plt.ylabel('Mean loss', fontsize=fontSize)
            plt.savefig(plotPath + original_string[:-3] + "_loss.pdf", bbox_inches="tight")
            plt.close()
            
            #Plot 3: test accuracy
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(df['Epoch'], df.iloc[:,4], ":*b", lw=0.5,label='EAMSGD')
            if isPlottingBenchmark:
                plt.plot(df_benchmark['Epoch'], df_benchmark['Test_Accuracy'], ":^r", lw=0.5,label='MSGD')
                plt.legend()
            plt.title(dataset + ' Test Accuracy: ' + new_string)
            plt.xlabel('Number of Epochs', fontsize=fontSize)
            plt.ylabel('Classification accuracy', fontsize=fontSize)
            plt.savefig(plotPath + original_string[:-3] + "_test_accuracy.pdf", bbox_inches="tight")
            plt.close()
            
            
            # except:
            #     pass
            
# def isCifar(title):
#     if title[:20] == 'training_stats_cifar':
#         return True

# def isMNIST(title):
#     if title[:20] != 'training_stats_cifar' and title[:14] == 'training_stats':
#         return True

def renameToAlpha(formatPattern, formattedString):
    folderPath = "../results/cifar/eamsgd/"
    newFolderPath = "../results/cifar/eamsgd_new/"
    
    for filename in os.listdir(folderPath):
        #resave beta files with alpha value
        original_string = filename
        numbers = re.findall(formatPattern, original_string)
        
        workers = int(numbers[0][0])-1
        tau = int(numbers[2][0])
        beta = int(numbers[3][0])
        alpha = beta/tau/workers
        new_string = re.sub(r'beta_\d+(\.\d+)?', 'alpha_' + str(alpha), original_string)

        if alpha == 0.25 or alpha == 0.125:
            print(folderPath + original_string)
            print(newFolderPath + new_string)
            shutil.copy(folderPath + original_string, newFolderPath + new_string)

def copyAllAlphaFiles():
    folderPath = "../results/cifar/eamsgd/"
    newFolderPath = "../results/cifar/eamsgd_new/"
    
    for filename in os.listdir(folderPath):
        if re.search(r"alpha", filename):
            print(filename)
            shutil.copy(folderPath + filename, newFolderPath + filename)
        
def communicationRatios(folderPath):
    taus = [2,4,8]
    workers = [2,4,8]
    communicationRatios = np.zeros((len(taus),len(workers)))
    print(communicationRatios.shape)
    for i in range(len(taus)):
        for j in range(len(workers)):
            tau = taus[i]
            worker = workers[j]
            
            #Table value
            for k in range(1,worker+1):
                workerFile = "stats_cifar_EAMSGD_size" + str(worker + 1) + "_rank_" + str(k) + "_tau_" + str(tau) + "_alpha_0.25_delta_0.9_momentum_0.txt"
                df = pd.read_csv(folderPath+ workerFile, sep = ",")
                communicationRatios[j,i] += 1/worker * df[' Total_comm_time'].iloc[-1]/df['Duration'].iloc[-1]
    
    return communicationRatios
    

def plotGridEAMSGD(folderPath,plotPath, df_benchmark):
    taus = [2,4,8]
    workers = [2,4,8]
    for i in range(len(taus)):
        for j in range(len(workers)):
            tau = taus[i]
            worker = workers[j]
            dataFile = "stats_cifar_EAMSGD_size" + str(worker + 1) + "_rank_0_tau_" + str(tau) + "_alpha_0.25_delta_0.9_momentum_0.txt"
            plotTitle = "$\\tau$ = " + str(tau) + ", p = " + str(worker)
            fileTitle = "EAMSGD_tau"+str(tau) + "_p"+str(worker)
            
            
            df = pd.read_csv(folderPath+ dataFile, sep = ",")
            df['Duration'] = df['Duration']/1000
            df = df.reset_index()
            df = df.rename(columns={"index": "Epoch"})
            df["Epoch"] = df["Epoch"] + 1
            
            #Plot 1: accuracy
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
        
            plt.plot(df['Epoch'], df.iloc[:,2], ":*b", lw=0.5,label='EAMSGD')
            plt.plot(df_benchmark['Epoch'], df_benchmark['Accuracy'], ":^r", lw=0.5,label='MSGD')
            plt.legend()
            plt.title(plotTitle)
            plt.xlabel('Number of Epochs', fontsize=fontSize)
            plt.ylabel('Training accuracy', fontsize=fontSize)
            plt.savefig(plotPath + fileTitle+ "_accuracy.pdf", bbox_inches="tight")
            plt.close()
            
            #Plot 2: loss
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(df['Epoch'], df.iloc[:,3], ":*b", lw=0.5,label='EAMSGD')
            plt.plot(df_benchmark['Epoch'], df_benchmark['Sample_Mean_Loss'], ":^r", lw=0.5,label='MSGD')
            plt.legend()
            plt.title(plotTitle)
            plt.xlabel('Number of Epochs', fontsize=fontSize)
            plt.ylabel('Mean training loss', fontsize=fontSize)
            plt.savefig(plotPath + fileTitle+ "_loss.pdf", bbox_inches="tight")
            plt.close()
            
            #Plot 3: test accuracy
            plt.figure()
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(df['Epoch'], df.iloc[:,4], ":*b", lw=0.5,label='EAMSGD')
            plt.plot(df_benchmark['Epoch'], df_benchmark['Test_Accuracy'], ":^r", lw=0.5,label='MSGD')
            plt.legend()
            plt.title(plotTitle)
            plt.xlabel('Number of Epochs', fontsize=fontSize)
            plt.ylabel('Test accuracy', fontsize=fontSize)
            plt.savefig(plotPath + fileTitle + "_test_accuracy.pdf", bbox_inches="tight")
            plt.close()
            
 
fontSize = 15
folder_path_CIFAR = "../results/CIFAR/"
plotPath = "../plots/"

pattern = r"(\d+(\.\d+)?)"
# Original string format
# original_string = "training_stats_cifar_eamsgd_size3_rank_0_tau_4_beta_3.96_delta_0.9_momentum_0.9._loss"
# original_string_EASGD = "training_stats_size3_rank_0_tau_8_beta_3.96"
# example_string_EASGD = "workers = 2, $\\tau$ = 5, $\\beta$ = 3.98,$\\delta$ = 0.9"
# formatted_string = re.sub(pattern, "{}", example_string_EASGD)
# print(formatted_string)

formattedStringMSGD = "$\\delta$ = {}"
formattedStringEASGD = "workers = {}, $\\tau$ = {}, $\\alpha$ = {}"
formattedStringEAMSGD = "workers = {}, $\\tau$ = {}, $\\alpha$ = {},$\\delta$ = {}"

#Plot MSGD
folderPathMSGD = folder_path_CIFAR + "msgd/"
#plotAllInFolder(folderPathMSGD,plotPath, False,None,True, formattedStringMSGD, pattern,True, fontSize)

# Target string format

# Regular expression pattern to match numbers

df_cifar_MSGD = pd.read_csv(folder_path_CIFAR + 'msgd/stats_cifar_MSGD_delta_0.9.txt')
df_cifar_MSGD['Duration'] = df_cifar_MSGD['Duration']/1000
df_cifar_MSGD = df_cifar_MSGD.reset_index()
df_cifar_MSGD = df_cifar_MSGD.rename(columns={"index": "Epoch"})
df_cifar_MSGD["Epoch"] = df_cifar_MSGD["Epoch"] + 1

#Plot all EAMSGD
folderPathEAMSGD = folder_path_CIFAR + "eamsgd/"
#plotAllInFolder(folderPathEAMSGD,plotPath, True,df_cifar_MSGD,True, formattedStringEAMSGD, pattern,True, fontSize)



#Plot all EASGD
# folderPathEASGD = folder_path_CIFAR + "easgd/"
# plotAllInFolder(folderPathEASGD,plotPath, True,df_cifar_MSGD,True, formattedStringEASGD, pattern,True, fontSize)

# df_mnist = pd.read_csv(folder_path + 'training_stats_sequential_batch_size_1000.txt')
# df_mnist['Duration'] = df_mnist['Duration']/1000
# df_mnist = df_mnist.reset_index()
# df_mnist = df_mnist.rename(columns={"index": "Epoch"})
# df_mnist["Epoch"] = df_mnist["Epoch"] + 1

#Time communication ratio 
communicationRatioTable = communicationRatios(folderPathEAMSGD)
print(communicationRatioTable*100)

#plotGridEAMSGD(folderPathEAMSGD,plotPath,df_cifar_MSGD)

























# # Loop over each file in the directory
# for filename in os.listdir(folder_path):
#     # Check if the entry is a file
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         #print(filename)  # Do whatever you want with the filename
#         try:
#             if isCifar(filename):
#                 dataset = "CIFAR10"
#                 df_benchmark = df_cifar
#             elif isMNIST(filename):
#                 print("hi")
#                 dataset = "MNIST"
#                 df_benchmark = df_mnist
#             #     print(filename)
#             #if isMNIST(filename):
#             #    print(filename)
#             else:
#                 continue
#             original_string = filename
#             numbers = re.findall(pattern, original_string)
#             del numbers[1]
#             del numbers[-1]
#             numbers[0] = (str(int(numbers[0][0])-1),'') #reduce workers by 1 (because of root)
#             # Replace numbers with placeholders

#             # Format the string with the extracted numbers
#             new_string = formatted_string.format(*[number[0] for number in numbers])

#             df = pd.read_csv(folder_path + filename)
#             df['Duration'] = df['Duration']/1000
#             df = df.reset_index()
#             df = df.rename(columns={"index": "Epoch"})
#             df["Epoch"] = df["Epoch"] + 1
            
#             #Plot 1: accuracy
#             plt.figure()
#             plt.rcParams.update({'font.size': fontSize})
#             plt.plot(df['Epoch'], df['Accuracy'], ":*b", lw=0.5,label='EAMSGD')
#             plt.plot(df_benchmark['Epoch'], df_benchmark['Accuracy'], ":^r", lw=0.5,label='MSGD')
#             plt.title(dataset + ' training accuracy: ' + new_string)
#             plt.xlabel('Number of Epochs', fontsize=fontSize)
#             plt.ylabel('Classification accuracy', fontsize=fontSize)
#             plt.legend()
#             plt.savefig(plots_path + original_string[:-3] + "_accuracy.pdf", bbox_inches="tight")
#             plt.close()
            
#             #Plot 2: loss
#             plt.figure()
#             plt.rcParams.update({'font.size': fontSize})
#             plt.plot(df['Epoch'], df['Sample_Mean_Loss'], ":*b", lw=0.5,label='EAMSGD')
#             plt.plot(df_benchmark['Epoch'], df_benchmark['Sample_Mean_Loss'], ":^r", lw=0.5,label='MSGD')
#             plt.title('Training loss: ' + new_string)
#             plt.xlabel('Number of Epochs', fontsize=fontSize)
#             plt.ylabel('Mean loss', fontsize=fontSize)
#             plt.legend()
#             plt.savefig(plots_path + original_string[:-3] + "_loss.pdf", bbox_inches="tight")
#             plt.close()
#             pass
#         except:
#             pass
        