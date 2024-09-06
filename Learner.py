import Class
class Learner: 

    def __init__(self, data, classPlace):
        self.data = data
        self.classPlace = classPlace
        self.train()
    
    def train(self):
        #train the model
        classes = []
        classesData = {}
        for index, row in self.data.iterrows():
            if row[self.classPlace] not in classes:
                classes.append(row[self.classPlace])
                classify = Class.Class(row[self.classPlace], 600)
                classesData[row[self.classPlace]] = classify
            classesData[row[self.classPlace]].add_data(row)
        print(classesData)
        print("******************")
        for c in classesData.keys():
            classesData[c].printData()
            print()
            classesData[c].createAttributes()
            print()