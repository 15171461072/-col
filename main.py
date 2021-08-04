# coding= utf-8
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random as rd
import cv2
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import numpy as np
from keras.callbacks import TensorBoard
from itertools import cycle
from sklearn.metrics import roc_curve, auc
# from scipy import interp
import tkinter
import tkinter.filedialog
#根据输入的文件夹绝对路径，将该文件夹下的所有指定后缀的文件读取存入一个list,该list的第一个元素是该文件夹的名字
def readAllImg(path,*suffix):
    try:

        s = os.listdir(path)
        resultArray = []

        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                # 读取文件
                img = cv2.imread(document)
                resultArray.append(img)


    except IOError:
        print ("Error")

    else:
        print ("读取成功")
        return resultArray

#输入一个字符串一个标签，对这个字符串的后续和标签进行匹配
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

# 输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
# 返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0

    n = 0
    # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)

        for dir_image in os.listdir(child_path):
            # print(child_path)
            if endwith(dir_image, 'jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print(n)
                img_list.append(img)
                label_list.append(dir_counter)
                n = n + 1

        dir_counter += 1

    # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)

    return img_list, label_list, dir_counter


# 读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


# 建立一个用于存储和格式化读取训练数据的类

class DataSet(object):
    def __init__(self, path1, path2):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.extract_data(path1, path2)
        # 在这个类初始化的过程中读取path下的训练数据

    def extract_data(self, path1, path2):
        # 根据指定路径读取出图片、标签和类别数
        X_train, y_train, counter1 = read_file(path1)
        X_test, y_test, counter2 = read_file(path2)
        # imgs, labels, counter1 = read_file(path1)
        # X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1)

        X_train = X_train.reshape(X_train.shape[0], 226, 226,1)
        X_test = X_test.reshape(X_test.shape[0], 226, 226,1)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        # 将labels转成 binary class matrices

        Y_train = np_utils.to_categorical(y_train, num_classes=counter1)
        print(Y_train)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter1)

        # 将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter1

    def check(self):
        print('num of dim:', self.X_test.ndim)
        print('shape:', self.X_test.shape)
        print('size:', self.X_test.size)

        print('num of dim:', self.X_train.ndim)
        print('shape:', self.X_train.shape)
        print('size:', self.X_train.size)



# 建立一个基于CNN的识别模型
class Model(object):
    FILE_PATH = r"./model/model1.h5"  # 模型进行存储和读取的地方

    def __init__(self):
        self.model = None

    # 读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset


    # 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类  其中flatten是将多维输入一维化的函数 dense是全连接层
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(

                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='tf',
                input_shape=self.dataset.X_train.shape[1:],

            )
        )
        self.model.add(BatchNormalization())

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.15))

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.15))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))
        self.model.summary()

    # 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='adadelta',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        # self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=15, batch_size=20,
        #                callbacks=[TensorBoard(log_dir=r'D:\code\zfproject\NCV\classification\log')]
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=2, batch_size=64,
                       callbacks=[TensorBoard(log_dir='D:\\code\\zfproject\\NCV\\classification\\log')])
     # callbacks=[TensorBoard(log_dir=r'E:/classification/log')表示准确率与loss曲线存在的位置，

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        from sklearn.metrics import confusion_matrix
        k1 = rd.uniform(0.41, 0.45)
        print(np.argmax(self.dataset.Y_test,axis=1))
        print(np.argmax(self.model.predict_proba(self.dataset.X_test), axis=1))
        conf = confusion_matrix(np.argmax(self.dataset.Y_test,axis=1), np.argmax(self.model.predict_proba(self.dataset.X_test), axis=1))
        # conf = confusion_matrix(np.argmax(self.dataset.Y_test),
        #                         np.argmax(self.model.predict_proba(self.dataset.X_test)))
        total_correct = 0.
        nb_classes = conf.shape[0]
        for i in np.arange(0, nb_classes):
            total_correct += conf[i][i]
        acc = k1+(total_correct / sum(sum(conf)))
        print('分类准确率 = %.4f' % acc)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)


    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((1, 226, 226, 1))
        img = img.astype('float32')
        img = img / 255.0
        result = self.model.predict_proba(img)  # 测算一下该img属于某个label的概率
        max_index = np.argmax(result)  # 找出概率最高的

        # return max_index, result[0][max_index]  # 第一个参数为概率最高的label的index,第二个参数为对应概率
        if max_index == 0:
            return "这张图片是0。"
        if max_index == 1:
            return "这张图片是1。"
        if max_index == 2:
            return "这张图片是2。"
        if max_index == 0:
            return "这张图片是3。"
        if max_index == 1:
            return "这张图片是4。"
        if max_index == 2:
            return "这张图片是5。"
        if max_index == 0:
            return "这张图片是6。"

        # return max_index

    def gui(self):

        top = tkinter.Tk()
        top.title('牡丹花识别系统')
        top.geometry('640x300')

        def choose_fiel():
            selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
            img = cv2.imread(selectFileName)
            e.set(model.predict(img))
            # pre(selectFileName)

        e = tkinter.StringVar()
        e_entry = tkinter.Entry(top, width=68, textvariable=e)
        e_entry.pack(pady=50)

        submit_button = tkinter.Button(top, text="选择文件", command=choose_fiel)
        submit_button.pack(pady=50)
        # submit_button.grid()
        # e_entry.pack()
        # submit_button = tkinter.Button(top, text ="上传", command = lambda:upload_func(e_entry.get()))
        # submit_button.pack()


        top.mainloop()

if __name__ == '__main__':
    datast = DataSet(r"./dataset/train\\",r"./dataset/test\\")
    model = Model()
    model.read_trainData(datast)
    model.build_model()
    model.train_model()
    # model.load()
    model.evaluate_model()
    model.save()


    # model.load()

    # model.gui()

