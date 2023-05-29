# k means
import math
import copy
import random

# elbow
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# visualization
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
# %matplotlib inline

# time
from time import time

import warnings
warnings.filterwarnings('ignore')



def calc_dist(A,B):
    '''
    calculate euclidean distance << 수정여지 있음 
    '''
    assert len(A) == len(B)

    sum = 0
    for a,b in zip(A,B):
        sum += (a-b)**2
    
    return math.sqrt(sum)


def calc_vector_mean(array):
    '''
    array : 각 클러스터에 할당된 벡터의 리스트
    각 차원별 평균
    '''
    summation = [0]*len(array[0])
    for row in array:
        for col in range(len(row)):
            summation[col] += row[col]
    
    for s in range(len(summation)):
        summation[s] = summation[s] / len(array)
    
    return summation


def calc_diff(A,B):
    '''
    calculate euclidean distance between two arrays (A,B)
    arguments : 
        A, B : same shape arrays
    returns : euclidean scalar value
    '''
    tol = 0
    for a,b in zip(A,B):
        tol += calc_dist(a,b)

    return tol


class Clustering(object):
    def __init__(self, max_iter=10, tol=1e-4):
        # __init__(self, k, max_iter=10, tol=1e-4)
        '''
        arguments:
            k : cluster number
            max_iter : maximum number of iteration 몇번하지...
            tol : update tolerance. centroids 업데이트가 이 값보다 작으면 멈춤
        '''
        self.k = -1
        self.thrsh = -1
        self.max_iter = max_iter
        self.tol = tol
        self.thrsh = [] #  100*n
        self.distances = []
        


    def optimal_k(self, input):
        '''
        find optimal elbow 최적의 군집수 찾기
        '''
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1,12))
        visualizer.fit(input)
        opt_k = visualizer.elbow_value_
        return opt_k

    def random_init(self, array):
        '''
        k-means++ 
        '''
        M = [] # 군집 중심 리스트. 값.
        indices = [] # 군집 중심 값에 대한 인덱스
        i = random.randint(0,len(array))

        M.append(array[i])
        indices.append(i)
        while len(M) < self.k:
            max_dist = -float('inf')
            max_index = -1
            for i in range(len(array)):
                avg_dist = 0
                if i in indices:
                    continue
                for j in range(len(M)):
                    dist = calc_dist(array[i], array[j])
                    avg_dist += dist # 거리 합
                avg_dist /= len(M) # 거리 평균
                if max_dist < avg_dist:
                    max_dist = avg_dist
                    max_index = i

            # M에 속한 중심값과의 평균거리 계산 후 평균거리가 가장 큰 데이터를 M에 추가

            M.append(array[max_index])
            indices.append(max_index)

        return M
    
    def fit(self, X):
        '''
        arguments :
            X : input array
        returns :
            각 벡터가 어느 클러스터에 속하는지 리스트형태로 반환 [0,1,2,2,2,1,1,0,0 ...]
            클러스터 중심 (x,y)쌍 리스트 [[x,y],[x,y]..]
        '''
        t1 = time()

        self.distances = [] # 중심과의 거리

        self.k = self.optimal_k(X)
        print("opt_k : ", self.k)

        
        for i in range(self.k):
            self.distances.append([]) # [[],[],[]]

        self.centroids = self.random_init(X)
        
        for iter in range(self.max_iter):
            print(f'{iter+1} iteration...')

            # distances 비우기
            for i in range(len(self.distances)):
                 self.distances[i].clear()

            # 군집 재할당시 
            self.thrsh.clear()
            # 군집 재할당
            self._cluster_assign(X)
            # centroid와의 거리의 최대값 추가
            self.thrsh.append(self._max_distance(self.distances))
            # 군집 중심 계산
            self._centroids_update(X)

            if calc_diff(self.prev_centroids, self.centroids)<self.tol:
                break

        t2 = time()
        print("KMeans train time: {:.3f} sec".format(t2-t1))

        return self.assignments, self.centroids
    
    def _cluster_assign(self,X):
        '''
        데이터별로 군집 번호를 할당
        거리가 가장 짧은 군집 중심에 군집을 할당
        '''
        self.assignments=[]
        
        for data in X:
            min_dist = float('inf')
            min_index = -1
            max_dist = -float('inf')
            
            for i, centroid in enumerate(self.centroids):
                dist = calc_dist(data, centroid)
                if dist<min_dist:
                    min_dist = dist
                    min_index = i
            self.distances[i].append(min_dist) # [소속 centroid][중심까지의 거리]
            self.assignments.append(min_index)


    def _centroids_update(self, X):
        '''
        할당된 군집에 대해 군집 중심을 업데이트
        '''
        self.prev_centroids = copy.deepcopy(self.centroids)

        for i in range(self.k):
            # cluster i에 소속된 데이터들의 인덱스
            data_indices = list(filter(lambda x : self.assignments[x] == i, range(len(self.assignments))))
            
            # 군집에 할당된 데이터가 없을때
            # 아무 데이터나 랜덤으로 하나 강제 할당
            if len(data_indices)==0:
                r = random.randint(0,len(X))
                self.centroids[i] = X[r]
                continue

            cluster_data = []
            for index in data_indices:
                cluster_data.append(X[index])
            self.centroids[i] = calc_vector_mean(cluster_data)
    
    def _max_distance(self, dist):
        '''
        # list : # [[],[],[]]
        # [소속 centroid][중심까지의 거리]
        return : [클러스터별 최대거리]
        '''        
        max_distances = []
        for i in range(len(dist)):
            max_distances.append(max(dist[i], default=0))
        return max_distances

        



# 일단 시각화 코드
def tsne_visualization(X):
    x_cluster = X # 100*100
    x_cluster = x_cluster.detach().cpu().numpy()
    standard = StandardScaler()
    x_std = standard.fit_transform(x_cluster) 
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(x_std)
    palette = sns.color_palette("bright", 10)
    sns.scatterplot(x_2d[:,0], x_2d[:,1], palette=palette)
    plt.show()
    plt.savefig('저장 경로**********************************수정필요')
    print("tsne2 saved")
    '''
    # 3차원 t-SNE 임베딩
    tsne_np = TSNE(n_components = 3).fit_transform(train_df)

    # numpy array -> DataFrame 변환
    tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1', 'component 2'])

    ---

    tsne = TSNE(n_components=2, perplexity = 44, random_state = 300) #44 || 32
    result = tsne.fit_transform(x_std)
    tsne_x = result[:,0] # 100
    tsne_y = result[:,1] # 100

    # sns.scatterplot(x=tsne_x, y=tsne_y, hue=clusters.labels_)
    sns.scatterplot(x=tsne_x, y=tsne_y)


    # centroid 시각화 코드
    print("centroids : ", centroids)
    centroids = np.transpose(centroids)
    print(centroids)

    plt.scatter(x = centroids[0], y = centroids[1], color = 'black')

    '''






def anomaly_detection(X, thrsh):
    #anomaly array
    anomaly_array = []
    
    #threshold = int(0.01 * len(data))
    threshold = thrsh

    #create array of final dataframe ...? 이건 머임
    final_arr = final_df.values

    #dictionary with cluster and count of number of points
    cluster_len = {}

    for s in range(len(final_arr)):
        if final_arr[s][-1] in cluster_len:
            cluster_len[final_arr[s][-1]] += 1
        else:
            cluster_len[final_arr[s][-1]] = 1

    for clster,len_val in cluster_len.items():
        #detect isolation points
        if len_val == 1:
            anomaly_array.append(final_dict[clster])
        #detect clusters with points less than threshold
        elif len_val < threshold:
            for values in final_dict[clster]:
                anomaly_array.append(values)

    anamoly_dict = {}
    for clu_key,clu_val in final_dict.items():
        anamoly_dict[clu_key] = [(np.mean(clu_val,axis = 0) + 2*np.std(clu_val,axis = 0)).tolist(),(np.mean(clu_val,axis = 0) - 2*np.std(clu_val,axis = 0)).tolist()]
        
    z = 0
    for key1 in set(anamoly_dict.keys()) & set(final_dict.keys()):
        for value1 in final_dict[key1]:
            #check if each point is within the mean ± 2*SD range
            if ~(np.less(np.array(value1),np.array(anamoly_dict[key1][z])).any()) and np.greater(np.array(np.array(value1)),anamoly_dict[key1][z+1]).all():
                anomaly_array.append(value1)
            else:
                continue

    anomalies_array = [[np.round(float(i), 2) for i in nested] for nested in anomaly_array]
    
    if len(anomalies_array) == 0:
        print("No anomalies")
    else:
        for anomaly in anomalies_array:
            print(anomaly)



'''
case 1.
# k means로 이상탐지하는 함수
1. 클러스터링 후 centroid 받아옴. 군집을 하나로 취급.... 흠....>균일한 대역폭에서는 가능하겠지만... 
2. 거리 = sqrt((x-center)**2) 모든 x와 유클리디안 거리를 구한다. 
3. order_index = argsort(distance, axis = 0)
    -argsort : 정렬 전의 인덱스를 정렬한 것을 리스트형태로 반환한다. 
    indexes = order_index[-5:] // 마지막 원소, 거리가 제일 긴 5개의 원소들의 인덱스
    values = x[indexes] // 이상으로 파악된 값들 
    * 이때 x, 거리, 뭐뭐 전부 np.array() 이어야 x[index], argsort 사용가능.
4. 이상 visualization - plt.scatter(indexes, values, color='r')

case 2.
# 잔차 계산
residuals = data - tsne.inverse_transform(embedding_vector)

# 이상값 탐지
threshold = 3 # 임계값 설정
outliers = np.argwhere(np.abs(residuals) > threshold)

임계값 > train set에서의 최대 거리



'''


# 이상 처리...>>AT 코드 다시 보기
# Mahalanobis distance
