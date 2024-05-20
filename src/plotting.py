import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

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

def threshholdTestAccuracy():
    threshhold = 0.7
    
    path_CIFAR = "../results/cifar/"
    path_CIFAR_MSGD = path_CIFAR + "msgd/stats_cifar_MSGD_delta_0.9.txt"
    
    df_benchmark = pd.read_csv(path_CIFAR_MSGD)
    df_benchmark = addEpochColumnToDF(df_benchmark)
    interpolated_epoch = getInterpolatedEpoch(threshhold,df_benchmark['Test_Accuracy'])
    print("Interpolated epoch: ", interpolated_epoch)
    
    taus = np.array([2,4,8])
    workers = np.array([2,4,8])
    epochs = np.zeros((len(workers),len(taus)))
    for i in range(len(taus)):
        for j in range(len(workers)):
            tau = taus[i]
            worker = workers[j]
            filename = "stats_cifar_EAMSGD_size" + str(worker + 1) + "_rank_0_tau_" + str(tau) + "_alpha_0.25_delta_0.9_momentum_0.txt"
            df = pd.read_csv(path_CIFAR + "eamsgd/" + filename)
            interpolated_ep = getInterpolatedEpoch(threshhold,df['Testing_accuracy'])
            epochs[j][i] = interpolated_ep
    paralell_speedup = np.divide(interpolated_epoch,epochs)
    parallel_efficiency = np.divide(paralell_speedup.T, workers+1).T
    
    print("Epochs (interpolated) to reach", threshhold, "test accuracy:")
    print(epochs)
    print("Parallel speedup")
    print(paralell_speedup)
    print("Parallel efficiency")
    print(parallel_efficiency)
    

def getInterpolatedEpoch(threshhold,df_col):
    idx = np.argmax(df_col >= threshhold)
    acc_upper = df_col.iloc[idx]
    acc_lower = df_col.iloc[idx-1]
    frac_idx = (threshhold-acc_lower)/(acc_upper-acc_lower)
    interpolated_idx = idx -1 + frac_idx
    return interpolated_idx + 1
        
def communicationRatios(folderPath):
    taus = [2,4,8]
    workers = [2,4,8]
    communicationRatios = np.zeros((len(workers),len(taus)))

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

def plotAllSDG(plotPath,fontSize):
    path_CIFAR = "../results/cifar/"
    path_MNIST = "../results/mnist/"
    
    path_CIFAR_SGD = path_CIFAR + "msgd/stats_cifar_MSGD_delta_0.txt"
    path_CIFAR_MSGD = path_CIFAR + "msgd/stats_cifar_MSGD_delta_0.9.txt"
    path_CIFAR_EASGD = path_CIFAR + "easgd/stats_cifar_EASGD_size5_rank_0_tau_4_alpha_0.25.txt"
    path_CIFAR_EAMSGD = path_CIFAR + "eamsgd/stats_cifar_EAMSGD_size5_rank_0_tau_4_alpha_0.25_delta_0.9_momentum_0.txt"
    
    path_MNIST_SGD = path_MNIST + "msgd/stats_mnist_MSGD_delta_0.txt"
    path_MNIST_MSGD = path_MNIST + "msgd/stats_mnist_MSGD_delta_0.9.txt"
    path_MNIST_EASGD = path_MNIST + "easgd/stats_mnist_EASGD_size5_rank_0_tau_4_alpha_0.25.txt"
    path_MNIST_EAMSGD = path_MNIST + "eamsgd/stats_mnist_EAMSGD_size5_rank_0_tau_4_alpha_0.25_delta_0.9_momentum_0.txt"
    
    df_cifar_SGD = pd.read_csv(path_CIFAR_SGD)
    df_cifar_MSGD = pd.read_csv(path_CIFAR_MSGD)
    df_cifar_EASGD = pd.read_csv(path_CIFAR_EASGD)
    df_cifar_EAMSGD = pd.read_csv(path_CIFAR_EAMSGD)
    df_mnist_SGD = pd.read_csv(path_MNIST_SGD)
    df_mnist_MSGD = pd.read_csv(path_MNIST_MSGD)
    df_mnist_EASGD = pd.read_csv(path_MNIST_EASGD)
    df_mnist_EAMSGD = pd.read_csv(path_MNIST_EAMSGD)
    
    dict_CIFAR = {
        'SGD': df_cifar_SGD,
        'MSGD': df_cifar_MSGD,
        'EASGD': df_cifar_EASGD,
        'EAMSGD': df_cifar_EAMSGD
    }
    dict_MNIST = {
        'SGD': df_mnist_SGD,
        'MSGD': df_mnist_MSGD,
        'EASGD': df_mnist_EASGD,
        'EAMSGD': df_mnist_EAMSGD
    }
    
    for key in dict_CIFAR:
        dict_CIFAR[key] = addEpochColumnToDF(dict_CIFAR[key])
    
    for key in dict_MNIST:
        dict_MNIST[key] = addEpochColumnToDF(dict_MNIST[key])
    
    colors = {
        'SGD': ':*b',
        'MSGD': ':^r',
        'EASGD': ':vg',
        'EAMSGD': ':sc'
    }
    
    #FIGURE 4
    plt.figure()
    plt.rcParams.update({'font.size': fontSize})

    for key in dict_CIFAR:
        df = dict_CIFAR[key]
        plt.plot(df['Duration'], df.iloc[:,2], colors[key], lw=0.5,label=key)
    plt.legend()
    plt.title("CIFAR: Test Accuracy vs. Wall-Clock Time")
    plt.xlabel('Wall-clock time (s)', fontsize=fontSize)
    plt.ylabel('Test accuracy', fontsize=fontSize)
    plt.savefig(plotPath + "allSGD_CIFAR"+ "_duration.pdf", bbox_inches="tight")
    plt.close()
    
    #FIGURE 5
    plt.figure()
    plt.rcParams.update({'font.size': fontSize})

    for key in dict_CIFAR:
        df = dict_CIFAR[key]
        plt.plot(df['Epoch'], df.iloc[:,2], colors[key], lw=0.5,label=key)
    plt.legend()
    plt.title("CIFAR: Test Accuracy")
    plt.xlabel('Number of Epochs', fontsize=fontSize)
    plt.ylabel('Test accuracy', fontsize=fontSize)
    plt.savefig(plotPath + "allSGD_CIFAR"+ "_epoch.pdf", bbox_inches="tight")
    plt.close()
    
    #MNIST over epochs
    plt.figure()
    plt.rcParams.update({'font.size': fontSize})

    for key in dict_MNIST:
        df = dict_MNIST[key]
        plt.plot(df['Epoch'], df.iloc[:,2], colors[key], lw=0.5,label=key)
    plt.legend()
    plt.title("MNIST: Test Accuracy")
    plt.xlabel('Number of Epochs', fontsize=fontSize)
    plt.ylabel('Test accuracy', fontsize=fontSize)
    plt.savefig(plotPath + "allSGD_MNIST"+ "_epoch.pdf", bbox_inches="tight")
    plt.close()


def plotGridEAMSGD(folderPath,plotPath, df_benchmark, fontSize):
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
            df = addEpochColumnToDF(df)
            
            #FIGURE 6
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
            
            #FIGURE 7
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
            
            #TRAINING LOSS - NOT USED
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
            
def addEpochColumnToDF(df):
    df['Duration'] = df['Duration']/1000
    df = df.reset_index()
    df = df.rename(columns={"index": "Epoch"})
    df["Epoch"] = df["Epoch"] + 1
    return df
    
def main():
    fontSize = 15
    folder_path_CIFAR = "../results/CIFAR/"
    plotPath = "../plots/"

    pattern = r"(\d+(\.\d+)?)"
    formattedStringMSGD = "$\\delta$ = {}"
    formattedStringEASGD = "workers = {}, $\\tau$ = {}, $\\alpha$ = {}"
    formattedStringEAMSGD = "workers = {}, $\\tau$ = {}, $\\alpha$ = {},$\\delta$ = {}"

    folderPathMSGD = folder_path_CIFAR + "msgd/"
    df_cifar_MSGD = pd.read_csv(folder_path_CIFAR + 'msgd/stats_cifar_MSGD_delta_0.9.txt')
    df_cifar_MSGD = addEpochColumnToDF(df_cifar_MSGD)

    folderPathEAMSGD = folder_path_CIFAR + "eamsgd/"
    folderPathEASGD = folder_path_CIFAR + "easgd/"


    #GENERATE TABLE 1
    communicationRatioTable = communicationRatios(folderPathEAMSGD)
    print("Communication ratios:")
    print(communicationRatioTable*100)

    #GENERATE TABLE 2-4
    threshholdTestAccuracy()
    
    #GENERATE FIGURES 4 & 5
    plotAllSDG(plotPath, fontSize)

    #GENERATE FIGURES 6 & 7
    plotGridEAMSGD(folderPathEAMSGD,plotPath,df_cifar_MSGD,fontSize)

    
    
    #ALL PLOTS - NOT INCLUDED IN REPORT
    #Plot MSGD
    #plotAllInFolder(folderPathMSGD,plotPath, False,None,True, formattedStringMSGD, pattern,True, fontSize)

    #Plot all EAMSGD    
    #plotAllInFolder(folderPathEAMSGD,plotPath, True,df_cifar_MSGD,True, formattedStringEAMSGD, pattern,True, fontSize)

    #Plot all EASGD
    # plotAllInFolder(folderPathEASGD,plotPath, True,df_cifar_MSGD,True, formattedStringEASGD, pattern,True, fontSize)

main()