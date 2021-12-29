import dataloader
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Neural_Network:
    def __init__(self, images, labels, batch_size, learning_rate):
        self.images = np.reshape(images, (len(images), 1, 784))
        self.labels = labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.test_cost = []
        self.train_cost = []
        self.test_cost_epoch = []
        self.train_cost_epoch = []
        self.rho = 0.9 # 0.99 

        # model parameter
        self.W1 = self.linear_layer(784, 512) # 784, 512
        self.W2 = self.linear_layer(512, 256) # 512, 256
        self.W3 = self.linear_layer(256, 10) # 256, 10
        self.b1 = self.linear_layer(self.batch_size, 512) # 8, 512
        self.b2 = self.linear_layer(self.batch_size, 256) # 8, 256
        self.b3 = self.linear_layer(self.batch_size, 10) # 8, 10
        self.vx_1 = 0
        self.vx_2 = 0
        self.vx_3 = 0

        # layer
        self.output_layer_1 = 0
        self.output_layer_2 = 0
        self.output_layer_3 = 0
        self.d_output_layer_1 = 0
        self.d_output_layer_2 = 0
        self.soft = 0

        #accuracy
        self.train_accuracy = 0
        self.test_accuracy = 0

        # visualization
        self.y_predic = []
        self.y_true = []

        # top 3 list
        self.map_list = []
        self.top_3 = []

    def train(self):
        input_images = np.zeros((self.batch_size, 784))
        input_labels = np.zeros((self.batch_size, 10)) 

        for epoch in range(80):
            data = dataloader.Dataloader(".", is_train = True, shuffle = True, batch_size=100)
            for i in range(0, int(len(self.images) / self.batch_size)): # 60000/8 = 7500번 iteration = 1 epoch 60000/100 = 600
                
                # using __getitem__()
                get_image, get_label = data.__getitem__(i)
                input_images = np.reshape(get_image, (self.batch_size, 784))
                input_labels = get_label
                self.forward_pass(input_images, input_labels, is_train=True)
                self.backward_pass(input_images, input_labels)

            temp_train = sum(self.train_cost) / (len(self.images) / self.batch_size)
            self.train_cost = []
            self.train_cost_epoch.append(round(temp_train, 2)) # temp는 np array이다! pop이 없음
            print(epoch, " 번째 epoch -> train loss : ", round(temp_train, 2), end = ' ')
            print("train_accuracy : ", round(self.train_accuracy/len(data.images) * 100, 2), "% ")
            
            # 1 epoch으로 학습된 모델 test 하기
            self.test()
            self.train_accuracy = 0
            self.test_accuracy = 0

            # confusion matrix 
            if epoch % 10 == 0:
                confusion = confusion_matrix(self.y_true, self.y_predic, normalize = "pred")
                plt.figure(figsize=(16,16))
                sns.heatmap(confusion, annot=True, cmap = 'Blues')
                plt.title("Normalized CONFUSION MATRIX : Neural_Network_3_layer")
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                plt.show()

            # show top 3 accuracy
            if epoch % 10 == 0 and epoch != 0:
                for i in range(10):
                    self.top_3.append(sorted(self.map_list[i].items(), reverse = True))

                for i in range(10):
                    print("picture:", i , "Top 3", end = " ")

                    for j in range(1, 4):
                        plt.subplot(1, 3, j)
                        percentage, image = self.top_3[i][j]
                        plt.imshow(image, cmap='Greys_r')
                        plt.axis('off')
                        print(round(percentage*100, 2), "%", end = " ")
                    plt.show()

            # train & test loss graph
            if epoch % 10 == 0 and epoch != 0:
                print("train_cost_epoch : ", self.train_cost_epoch)
                print("test_cost_epoch : ", self.test_cost_epoch)
                plt.plot(range(0, epoch+1), self.train_cost_epoch, 'b', label='train')
                plt.plot(range(0, epoch+1), self.test_cost_epoch, 'r', label='test')
                plt.ylabel('Cost')
                plt.xlabel('Epochs')
                plt.legend(loc='upper right')
                plt.show()

    def test(self):
        data = dataloader.Dataloader(".", is_train = False, shuffle = True, batch_size=100) 
        for i in range(0, int(len(data.images) / self.batch_size)):
            get_image, get_label = data.__getitem__(i)
            input_images = np.reshape(get_image, (self.batch_size, 784))
            input_labels = get_label
            self.forward_pass(input_images, input_labels, is_train = False)
        
        temp_test = sum(self.test_cost) / (len(data.images) / self.batch_size)
        self.test_cost_epoch.append(round(temp_test, 2))
        self.test_cost = []
        print(".. 번째 epoch -> test  loss : ", round(temp_test, 2), end=' ')
        print("test accuracy : ", round(self.test_accuracy/len(data.images) * 100, 2), "%")
        

    def forward_pass(self, input_images, input_labels, is_train):
        if is_train == True:
            # 1st layer
            hidden_layer_1 = np.dot(input_images, self.W1) + self.b1 # hidden_layer_1 = 8*512 
            # ReLU
            self.output_layer_1 = self.ReLU(hidden_layer_1)
            self.d_output_layer_1 = self.derivation_ReLU(self.output_layer_1)
            
            # Leaky ReLU
            # self.output_layer_1 = self.Leaky_ReLU(hidden_layer_1)
            # self.d_output_layer_1 = self.derivation_Leaky_ReLU(self.output_layer_1)

            # 2nd layer
            hidden_layer_2 = np.dot(self.output_layer_1, self.W2) + self.b2 # hidden_layer_2 = 8*256
            # ReLU
            self.output_layer_2 = self.ReLU(hidden_layer_2)
            self.d_output_layer_2 = self.derivation_ReLU(self.output_layer_2)

            # Leaky ReLU
            # self.output_layer_2 = self.Leaky_ReLU(hidden_layer_2)
            # self.d_output_layer_2 = self.derivation_Leaky_ReLU(self.output_layer_2)
            
            # 3rd layer -> no relu
            self.output_layer_3 = np.dot(self.output_layer_2, self.W3) + self.b3 # hidden_layer_3 = 8*10
            #d_output_layer_3 = self.derivation_ReLU(output_layer_3)
            
            # softmax
            self.soft = self.Softmax(self.output_layer_3)
            
            # loss
            loss = self.Cross_Entropy_Loss(self.soft, input_labels) #/ self.batch_size #흠.. 이걸 왜 나눠주지?
            #self.cost.append(sum(loss)/self.batch_size)
            self.train_cost.append(loss/self.batch_size)

            # calculate accuracy
            for i in range(self.batch_size):
                prediction = np.argmax(self.soft[i])
                label_answer = np.argmax(input_labels[i])
                #self.y_predic.append(prediction)
                #self.y_true.append(label_answer)
                if prediction == label_answer:
                    self.train_accuracy += 1

        elif is_train == False:
            # 1st layer
            hidden_layer_1 = np.dot(input_images, self.W1) + self.b1
            output_layer_1 = self.ReLU(hidden_layer_1)
            d_output_layer_1 = self.derivation_ReLU(output_layer_1)

            # 2nd layer
            hidden_layer_2 = np.dot(output_layer_1, self.W2) + self.b2 # hidden_layer_2 = 8*256
            output_layer_2 = self.ReLU(hidden_layer_2)
            d_output_layer_2 = self.derivation_ReLU(output_layer_2)

            # 3rd layer -> relu 없음
            output_layer_3 = np.dot(output_layer_2, self.W3) + self.b3 # hidden_layer_3 = 8*10

            # softmax
            soft = self.Softmax(output_layer_3)
            loss = self.Cross_Entropy_Loss(soft, input_labels)
            self.test_cost.append(loss/self.batch_size)

            # calculate accuracy & top 3 accuracy
            for i in range(self.batch_size):
                prediction = np.argmax(soft[i]) # prediction은 index 이다!!
                label_answer = np.argmax(input_labels[i])
                self.y_predic.append(prediction)
                self.y_true.append(label_answer)
                
                if prediction == label_answer:
                    self.test_accuracy += 1    
                    for _ in range(10):
                        temp_map = {}
                        self.map_list.append(temp_map) # map_list = [] 안에는 dictionary로 구성되어 있다
                    temp_image = np.reshape(input_images[i]*255, (28, 28))
                    self.map_list[label_answer][soft[i][prediction]] = temp_image

    def backward_pass(self, input_images, input_labels):
        # back propagation
        # dW3 -> 256 * 10
        d_softmax_cross = self.derivation_softmax_cross(self.soft, input_labels) # d_softmax_cross = 8*10
        dW3 = np.dot(d_softmax_cross.T, self.output_layer_2)

        # dW2 -> 256 * 512
        dw2_temp = np.dot(self.W3, d_softmax_cross.T) # W3 = 256*10, d_softmax_cross = 8*10, result = 256*8
        dw2_temp2 = dw2_temp.T * self.d_output_layer_2 # temp.T = 8*256 d_output_layer_2 = 8*256, result= 8*256
        dW2 = np.dot(dw2_temp2.T, self.output_layer_1) # temp2.T = 256*8, output_layer_1 = 8*512, result= 256*512 

        # dW1 -> 512 * 784
        dw1_temp = np.dot(self.W2, dw2_temp2.T) # W2 = 512*256, dw2_temp2 = 8*256, result = 512*8
        dw1_temp2 = dw1_temp * self.d_output_layer_1.T # dw3_temp = 512*8, d_output_layer_1 = 8*512, result = 512*8
        dW1 = np.dot(dw1_temp2, input_images) # dw3_temp2 = 512*8, input_images = 8*784, result = 512*784

        # db3 -> 8*10
        db3 = d_softmax_cross 

        # db2 -> 8*256
        db2 = dw2_temp2

        # db1 -> 8*512
        db1 = dw1_temp2.T

        # SGD + momentum 
        self.vx_1 = self.rho * self.vx_1 - self.learning_rate * dW1.T
        self.W1 = self.W1 + self.vx_1
        self.vx_2 = self.rho * self.vx_2 - self.learning_rate * dW2.T
        self.W2 = self.W2 + self.vx_2
        self.vx_3 = self.rho * self.vx_3 - self.learning_rate * dW3.T
        self.W3 = self.W3 + self.vx_3
    
        # W1 = W1 - self.learning_rate * dW1.T # W1 = 784*512
        # W2 = W2 - self.learning_rate * dW2.T
        # W3 = W3 - self.learning_rate * dW3.T

        # b1 = b1 - self.learning_rate * db1
        # b2 = b2 - self.learning_rate * db2
        # b3 = b3 - self.learning_rate * db3

    def linear_layer(self, row, column):
        np.random.seed(0)
        linear_layer = np.random.randn(row, column)
        return linear_layer

    def momoentun(self, row, column):
        return np.zeros(row, column)

    def ReLU(self, matrix):
        #return np.maximum(0, x)
        matrix[matrix<0]=0
        return matrix

    def derivation_ReLU(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def Leaky_ReLU(self, x):
        return np.maximum(0.01*x, x)

    def derivation_Leaky_ReLU(self, x):
        dRelu_dx = x
        dRelu_dx[dRelu_dx < 0] = 0.01
        dRelu_dx[dRelu_dx > 0] = 1
        return dRelu_dx

    def Elu(self, x, alp=0.5):
        return (x>0)*x + (x<=0)*(alp*(np.exp(x) - 1))

    def Softmax(self, x): # x : 8*10
        s = np.exp(x)
        total = np.sum(s, axis=1).reshape(-1,1)
        return s/total

    def Cross_Entropy_Loss(self, softmax_matrix, label_matrix):
        # delta -> very small value (if y is 0, then it can prevent -inf) 
        delta = 1e-7 
        return -np.sum(label_matrix*np.log(softmax_matrix+delta))


    def derivation_softmax_cross(self, softmax_matrix, label_matrix):
        return softmax_matrix - label_matrix

if __name__ == "__main__":
    data_load = dataloader.Dataloader(".", is_train = True, shuffle=True, batch_size=100)
    data = Neural_Network(data_load.images, data_load.labels, data_load.batch_size, learning_rate = 0.0001)
    data.train()

