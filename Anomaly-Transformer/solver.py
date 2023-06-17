import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment




from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
# from yellowbrick.cluster import KElbowVisualizer
# from yellowbrick.text import TSNEVisualizer
import seaborn as sns
from numpy import sqrt, random, array, argsort
# from sklearn.preprocessing import RobustScaler

# import kmeansAd as ka

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler
import collections




assc_disc_num = 0

def multi_mean(array):
    # array : series[0][0].shape = 8*100*100 = array.shape
    mean = torch.zeros(array[0].shape).cuda()
    for i in range(len(array)):
        mean += array[i]
    mean = mean/len(array)

    return mean # 100*100

def perf_measure(y_actual, y_hat):
    '''
    sensitivity  = TP / (TP+FN)
    specificity  = TN / (TN+FP)
    pos_pred_val = TP/ (TP+FP)
    neg_pred_val = TN/ (TN+FN)
    '''

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    recall  = TP / (TP+FN)
    precision = TP/ (TP+FP)
    false_pos_rate = FP/(TN+FP)
    accuracy = (TP+TN) / (TP+FP+TN+FN)

    return recall, precision, false_pos_rate, accuracy


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.criterion = nn.MSELoss(reduction='none')
        self.criterion = nn.MSELoss()
        # self.eps =  -1
        # self.min_samples = -1

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        # print("loss1 len :",len(loss_1)) 28
        # print("loss2 len :",len(loss_2)) 28 ass disc 횟수랑 동일함

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        '''
        # #############################################################################################################################################################
        cluster_scaler = StandardScaler()
        cluster_scaler2 = MinMaxScaler()
        eps = 0.05 # 조정 필요
        min_samples = 4
        dbscan = DBSCAN(eps=eps, min_samples=min_samples) #  metric='precomputed'
        tsne_model = TSNE(n_components=2,learning_rate=300)
        neighbors = NearestNeighbors(n_neighbors=20)
        '''


        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):


                self.optimizer.zero_grad()
                iter_count += 1
                #print("input data : ", input_data.shape) 256*100*38
                #print("label : ", labels.shape) 256*100
                input = input_data.float().to(self.device)
                #print("input :", input.shape)256*100*38
                
      


                output, series, prior, _ = self.model(input)
                # print("output : ", output.shape) 256*100*38
                # print("series : ", len(series))
                # print("prior : ", len(prior)) 3




                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                #print("prior length ************* :",len(prior)) 3
                # len(series) = 3
                # len(series[0]) = 256
                # len(series[0][0]) = 8
                # len(series[0][0][0]) = 100
                # len(series[0][0][0][0]) = 100
                # series[0].shape = 256*8*100*100
                # series[0][0].shape = 8*100*100
                # print("series :", series[0][0][0])
                #print("prior shape ************* :",prior[1].shape) 258*8*100*100
                # testtest = series[0][0]
                # print("test shape : ",testtest[0].shape) series[0][0].shape

                


                '''
                # ######################################################################################################################################################
                cluster_data = multi_mean(series) # encoder 값들 간 평균 256*8*100*100
                for sampleNum in range(len(cluster_data)):
                    for headNum in range(len(cluster_data[sampleNum])):
                        cluster_data_mean = multi_mean(cluster_data[headNum]) # multi head 값들 간 평균 100*100
                        cluster_data_mean = cluster_data_mean.detach().cpu().numpy()
                    print("cluster data : ", cluster_data.shape)
                    print("cluster data mean : ", cluster_data_mean.shape)
                    print("sampleNum : {}".format(sampleNum))
                    cluster_scaler2.fit(cluster_data_mean)
                    cluster_scaled = cluster_scaler2.transform(cluster_data_mean)

                    # stdDf = pd.DataFrame(cluster_scaled)
                    
                    neighbors_fit = neighbors.fit(cluster_scaled)
                    distances, indices = neighbors_fit.kneighbors(cluster_scaled)

                    distances = np.sort(distances, axis=0)
                    distances = distances[:,1]
                    plt.plot(distances)
                    plt.savefig("/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/img2/"+str(sampleNum)+"optimaleps.png")
                    print("plot saved")
                    
                    min_samples = range(4,15)
                    eps = np.arange(0.5,1.5, 0.01) # returns array of ranging from 0.05 to 0.13 with step of 0.01

                    output = []

                    for ms in min_samples:
                      for ep in eps:
                        labels = DBSCAN(min_samples=ms, eps = ep).fit(cluster_scaled).labels_
                        score = silhouette_score(cluster_scaled, labels)
                        output.append((ms, ep, score))

                    min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1]
                    print(f"Best silhouette_score: {score}")
                    print(f"min_samples: {min_samples}")
                    print(f"eps: {eps}")


                    

                    clusters = dbscan.fit(cluster_scaled)
                    labels = dbscan.labels_
                        
                    transformed_tsne = tsne_model.fit_transform(cluster_scaled)
                    # tsne_array = np.array(transformed_tsne)

                    palette = sns.color_palette("bright")
                    sns.scatterplot(x=transformed_tsne[:,0], y=transformed_tsne[:,1], hue=labels, palette=palette)
                        
                    plt.savefig("/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/img2/"+str(sampleNum)+".png")
                    print("plot saved")
                    plt.clf()


                        
                        
                        # sil_score = metrics.silhouette_score(cluster_scaled, labels, metric='euclidean')
                        
                    print("min samples : {} | eps : {} ".format(min_samples, eps))
                        # print("silhouette score : {}".format(sil_score))


                        # neighbors = NearestNeighbors(n_neighbors=20)
                        # neighbors_fit = neighbors.fit(cluster_data_mean)
                        # distances, indices = neighbors_fit.kneighbors(cluster_data_mean)

                        # distances = np.sort(distances, axis=0)
                        # distances = distances[:,1]
                        # plt.figure(figsize=(20, 10))
                        # plt.plot(distances)
                        # plt.savefig('/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/img/cluster_elbow.png')

                    prtnum = 0
                    while(-1 in labels):
                      eps += 0.01
                      dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                      clusters = dbscan.fit(cluster_scaled)
                      labels = dbscan.labels_
                          # sil_score = metrics.silhouette_score(cluster_scaled, labels, metric='euclidean')
                      prtnum += 1
                      if(prtnum // 10) == 1:
                        transformed_tsne = tsne_model.fit_transform(cluster_scaled)
                    # tsne_array = np.array(transformed_tsne)

                        palette = sns.color_palette("bright")
                        sns.scatterplot(x=transformed_tsne[:,0], y=transformed_tsne[:,1], hue=labels, palette=palette)
                        
                        plt.savefig("/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/img2/"+str(sampleNum)+".png")
                        print("plot saved")
                        plt.clf()
                        prtnum = 0
                        
                      print("min samples : {} | eps : {} | 클러스터 레이블 : \n{}".format(min_samples, eps, labels))
                          # print("silhouette score : {}".format(sil_score))

                        

                        
                print("final eps :{}".format(eps)) # num = 4, eps = 0.223
                # epoch 다 돌리니 eps = 0.953... 중간에 뭔가 잘못했나 
                self.eps = eps # min sample = 4 / eps = 3.37
                '''

                



                for u in range(len(prior)):
                    # print("u : ", u) 0, 1, 2
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                # print("where am i")
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
  

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        cluster_scaler = StandardScaler()
        cluster_scaler2 = MinMaxScaler()
        eps = 2.9 
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        tsne_model = TSNE(n_components=2,learning_rate=300)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []

        total_labels = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            cluster_data = multi_mean(series) 

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            
            for sampleNum in range(len(cluster_data)):
                for headNum in range(len(cluster_data[sampleNum])):
                    cluster_data_mean = multi_mean(cluster_data[headNum]) # multi head 값들 간 평균 100*100
                    cluster_data_mean = cluster_data_mean.detach().cpu().numpy()

                cluster_scaler2.fit(cluster_data_mean)
                cluster_scaled = cluster_scaler2.transform(cluster_data_mean)

                clusters = dbscan.fit_predict(cluster_scaled)
                cluster_labels = dbscan.labels_
                # labels = labels.detach().cpu().numpy()
                total_labels.append(cluster_labels)

                '''
                if (-1 in cluster_labels):
                # 시각화       
                  transformed_tsne = tsne_model.fit_transform(cluster_scaled)
                  tsne_array = np.array(transformed_tsne)

                  palette = sns.color_palette("bright")
                  sns.scatterplot(x=transformed_tsne[:,0], y=transformed_tsne[:,1], hue=cluster_labels, palette=palette)
                        
                  plt.savefig("/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/img2/"+str(sampleNum)+".png")
                  print("plot saved")
                  plt.clf()
                '''
                '''
                # 실루엣 score : 0.5674
                if (-1 in cluster_labels):
                   sil_score = silhouette_score(cluster_scaled, cluster_labels, metric='euclidean')
                   print("silhouette score : {}".format(sil_score))
                '''

                '''   
                  
                # sil_score = metrics.silhouette_score(cluster_scaled, labels, metric='euclidean')
                        
                # print("silhouette score : {}".format(sil_score))

                        
                #print("min samples : {} | eps : {} | 클러스터 레이블 : \n{}".format(min_samples, eps, labels))
                # print("silhouette score : {}".format(sil_score))

                        
                '''

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

            ############################################################################


        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        total_labels = np.concatenate(total_labels, axis=0).reshape(-1)
        print("total labels : ", total_labels.shape)
        total_labels = np.where(total_labels==-1, 1, 0)
        # total_labels = np.where(total_labels>=1, 0)
        np.savetxt('/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/totalLabels.txt', total_labels, fmt = '%d', delimiter = ' ', header='total Labels')  
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)



        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        np.savetxt('/content/drive/MyDrive/Colab_Project/Anomaly-Transformer/predLabels.txt', pred, fmt = '%d', delimiter = ' ', header='pred Labels')  

        compare_arr = (pred == total_labels)

        
        
        thresh = (94.69 / 99.29) * thresh # 0.02 증가
        
        not_equal = (pred != total_labels)
        not_equal_index = np.where(not_equal == 0)
        not_equal_index = not_equal_index[0]
        for v in not_equal_index:
          if(test_energy[v]>thresh):
            pred[v] = 1
        print("pred 갱신 완료") 
        # Accuracy : 0.9929, Precision : 0.8915, Recall : 0.9431, F-score : 0.9166
        
        
        

        
        count_true = sum(compare_arr)
        count_one = collections.Counter(total_labels)[1]
        print("at, cluster 일치하는 수 : ",count_true)
        print("AT와 cluster 일치율 : ", count_true/len(total_labels)*100)
        print("cluster anomaly 개수: ", count_one)
        count_one = collections.Counter(pred)[1]
        print("pred anomaly 개수 : ", count_one)

        pred_one = np.where(pred==1)
        pred_one = pred_one[0] # 1인 인덱스
        cluster_one = np.where(total_labels==1) # 1인 인덱스
        cluster_one = cluster_one[0]

        both = np.intersect1d(pred_one,cluster_one)
        print("1 중 겹치는 수 : ", both.size)

        pred_zero = np.where(pred==0)
        pred_zero = pred_zero[0] # 0인 인덱스
        cluster_zero = np.where(total_labels==0) # 0인 인덱스
        cluster_zero = cluster_zero[0]

        both = np.intersect1d(pred_zero,cluster_zero)
        print("0 중 겹치는 수 : ", both.size)
        
        


        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))


        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # clustering 
        accuracy = accuracy_score(gt, total_labels)*100
        precision = precision_score(gt, total_labels)*100
        recall = recall_score(gt, total_labels)*100
        f_score = f1_score(gt, total_labels)*100


        recall, precision, false_pos_rate, accuracy = perf_measure(gt, total_labels)
        print("Clustering false positive rate : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}".format(false_pos_rate*100,precision*100,recall*100))

        return accuracy, precision, recall, f_score

