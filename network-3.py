import numpy

#激活函数
def sigmoid (x):
    return 1/(1+numpy.exp(-x))

#求导
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#损失函数
#返回真实值减预测值平方的平均值
def mse_loss(y_tr,y_pre):
    return((y_tr - y_pre)**2).mean()


class nerualnetwo():
    #高斯分布随机数值
    def __init__(self):
       self.w1 = numpy.random.normal()
       self.w2 = numpy.random.normal()
       self.w3 = numpy.random.normal()
       self.w4 = numpy.random.normal()
       self.w5 = numpy.random.normal()
       self.w6 = numpy.random.normal()
       self.b1 = numpy.random.normal()
       self.b2 = numpy.random.normal()
       self.b3 = numpy.random.normal()
    #神经元的定义
    def feedforward(self,x):
        h1 = x[0]*self.w1+x[1]*self.w2+self.b1
        h1f = sigmoid(h1)
        h2 = x[0]*self.w3+x[1]*self.w4+self.b2
        h2f = sigmoid(h2)
        o1 = h1*self.w5+h2*self.w6+self.b3
        of = sigmoid(o1)
        return h1,h1f,h2,h2f,o1,of
    def simulate (self,x):
        h1 = x[0]*self.w1+x[1]*self.w2+self.b1
        h1f = sigmoid(h1)
        h2 = x[0]*self.w3+x[1]*self.w4+self.b2
        h2f = sigmoid(h2)
        o1 = h1*self.w5+h2*self.w6+self.b3
        of = sigmoid(o1)
        return of
    def train(self,data,all_y_tr):
        #设置循环次数为1000
        epochs = 1000
        #学习率为0.1
        learn_rate = 0.1
        #将以下运算做一个1000次的循环
        for i in range(epochs):
            #将输入数据与输出数据打包一一对应，然后分别赋值给x和y_tr
            for x , y_tr in zip(data,all_y_tr):
                #所有节点值组Valcell等于feedforward函数得出的结果
                valcell = self.feedforward(x)
                #y_pre（y预测值）为valcell的第六个值
                y_pre = valcell[5]
                #损失函数对y预测求偏导数
                der_L_y_pre = -2*(y_tr-y_pre)
                #y_预测对h1求编导数
                der_y_pre_h1 = der_sigmoid(valcell[4])*self.w5
                #y_预测对h2求偏导数
                der_y_pre_h2 = der_sigmoid(valcell[4])*self.w6
                #h1对w1求偏导数
                #以此类推
                der_h1_w1 = der_sigmoid(valcell[0])*x[0]
                der_h1_w2 = der_sigmoid(valcell[0])*x[1]
                der_h2_w3 = der_sigmoid(valcell[2])*x[0]
                der_h2_w4 = der_sigmoid(valcell[2])*x[1]
                der_y_pre_w5 = der_sigmoid(valcell[4])*valcell[1]
                der_y_pre_w6 = der_sigmoid(valcell[4])*valcell[3]
                der_y_pre_b3 = der_sigmoid(valcell[4])
                der_h1_b1 = der_sigmoid(valcell[0])
                der_h2_b2 = der_sigmoid(valcell[2])

                self.w1 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_w1
                self.w2 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_w2
                self.w3 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_w3
                self.w4 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_w4
                self.w5 -= learn_rate * der_L_y_pre * der_y_pre_w5
                self.w6 -= learn_rate * der_L_y_pre * der_y_pre_w6
                self.b1 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_b1
                self.b2 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_b2
                self.b3 -= learn_rate * der_L_y_pre *der_y_pre_b3
                if i % 10 ==0 :
                    y_pred = numpy.apply_along_axis(self.simulate,1,data)
                    loss = mse_loss (all_y_tr , y_pred)
                    print(i,loss)

#训练网络
data = numpy.array([[-2, -1],[25, 6],[17, 4],[-15, -6]])
all_y_trues = numpy.array([1,0,0,1])
ner = nerualnetwo()
ner.train(data,all_y_trues)

#预测：结果为0.99左右
test = numpy.array([-7, -3])
print(ner.simulate(test))
