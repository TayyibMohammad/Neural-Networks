import random
import math   

class LayerDense():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs=n_inputs
        self.n_outputs=n_outputs
        self.weights=[]
        self.biases=[]
        for i in range(n_inputs):
            weight = []
            for j in range(n_outputs):
                weight.append(random.randrange(-100, 100)/100)
            self.weights.append(weight)
        
        for i in range(n_outputs):
            self.biases.append(0)
            
        # return [self.weights, self.biases]
        
    def getParams(self):
        return [self.weights, self.biases]
            
    def forward(self, inputs, wts, bia):
        outputs=[]
        for i in range(len(wts[0])):
            output=0
            for j in range(len(wts)):
                output+=wts[j][i]*inputs[j]
            output+=bia[i]
            outputs.append(output)
        return outputs
    
    def resetParams(self, error):
        new_weights=[]
        new_biases=[]
        for i in range(self.n_inputs):
            new_weight=[]
            for j in range(self.n_outputs):
                new_weight.append(self.weights[i][j]+error*random.randrange(-100, 100)/100)
            new_weights.append(new_weight)
                
        for i in range(self.n_outputs):
            new_biases.append(self.biases[i]+error*random.randrange(-100, 100)/100)
            
        return [new_weights, new_biases]
    

    
class ActivationRelu():
    def forward(self, inputs):
        outputs=[]
        for i in range(len(inputs)):
            outputs.append(max(0, inputs[i]))
        return outputs
    
class SoftmaxActivation():
    def forward(self, inputs):
        outputs=[]
        for i in range(len(inputs)):
            outputs.append(math.exp(inputs[i]))
        total=0
        for i in range(len(outputs)):
            total+=outputs[i]
        for i in range(len(outputs)):
            outputs[i]/=total
        return outputs
    
class SigmoidActivation():
    def forward(self, inputs):                        
        outputs=[]
        for i in range(len(inputs)):
            outputs.append(1/(1+math.exp(-inputs[i])))
        return outputs
    
    
    
layer1=LayerDense(2, 3)
layer1_wts=layer1.getParams()[0]
layer1_bia=layer1.getParams()[1]

layer2=LayerDense(3, 3)
layer2_wts=layer2.getParams()[0]
layer2_bia=layer2.getParams()[1]

layer3=LayerDense(3, 4)
layer3_wts=layer3.getParams()[0]
layer3_bia=layer3.getParams()[1]


# ***************Data_Generation****************
X=[]
for i in range(-10, 10):
    for k in range(-10, 10):
        if(i!=0 and k!=0):
            coord=[]
            coord.append(i)
            coord.append(k)
            X.append(coord)
y=[]
for coord in X:
    if coord[0]>0 and coord[1]>0:
        y.append([1,0,0,0])
        
    elif coord[0]<0 and coord[1]>0:
        y.append([0,1,0,0]) 
    
    elif coord[0]<0 and coord[1]<0:
        y.append([0,0,1,0]) 
    
    elif coord[0]>0 and coord[1]<0:
        y.append([0,0,0,1]) 
        
    
# ***************Data_Generation_Ended**************





def calc(layer1_wts, layer1_bia, layer2_wts, layer2_bia, layer3_wts, layer3_bia):
    y_pred=[]
    i=0
    for i in range(len(X)):
        output=layer1.forward(X[i], layer1_wts, layer1_bia)
        output2=layer2.forward(ActivationRelu().forward(output), layer2_wts, layer2_bia)
        output3=layer3.forward(ActivationRelu().forward(output2), layer3_wts, layer3_bia)

        output4=SoftmaxActivation().forward(output3)
        y_pred.append(output4)
        i+=1
    return y_pred
    
    
def CalculateError(y, y_pred):
    error=0
    for i in range(len(y_pred)):
        for j in range(4):
            error+=(y[i][j]*math.log(y_pred[i][j]))/len(y_pred)
        
    return -1*error


# print(CalculateError(y, y_pred))

output=layer1.forward([60, -16.598], layer1_wts, layer1_bia)
output2=layer2.forward(ActivationRelu().forward(output), layer2_wts, layer2_bia)
output3=layer3.forward(ActivationRelu().forward(output2), layer3_wts, layer3_bia)
output4=SoftmaxActivation().forward(output3)

print(output4)




n_iteration=10_000
for i in range(n_iteration):
    y_pred=calc(layer1_wts, layer1_bia, layer2_wts, layer2_bia, layer3_wts, layer3_bia)   
    minError=CalculateError(y, y_pred)
    
    new_layer1_wts=layer1.resetParams(minError)[0]
    new_layer1_bia=layer1.resetParams(minError)[1]
    y_pred=calc(new_layer1_wts, new_layer1_bia, layer2_wts, layer2_bia, layer3_wts, layer3_bia)
    currError = CalculateError(y, y_pred)
    if(currError<minError):
        minError=currError
        layer1_wts=new_layer1_wts
        layer1_bia=new_layer1_bia
        print(minError," ",i)
        
        
        
    new_layer2_wts=layer2.resetParams(minError)[0]
    new_layer2_bia=layer2.resetParams(minError)[1]
    y_pred=calc(layer1_wts, layer1_bia, new_layer2_wts, new_layer2_bia, layer3_wts, layer3_bia)
    
    currError=CalculateError(y, y_pred)
    if(currError<minError):
        minError=currError
        layer2_wts=new_layer2_wts
        layer2_bia=new_layer2_bia
        print(minError," ",i)

        
        
        
    new_layer3_wts=layer3.resetParams(minError)[0]
    new_layer3_bia=layer3.resetParams(minError)[1]
    y_pred=calc(layer1_wts, layer1_bia, layer2_wts, layer2_bia, new_layer3_wts, new_layer3_bia)
    currError=CalculateError(y, y_pred)
    if(currError<minError):
        minError=currError
        layer3_wts=new_layer3_wts
        layer3_bia=new_layer3_bia
        print(minError," ",i)

         
         


output=layer1.forward([60, -16.598], layer1_wts, layer1_bia)
output2=layer2.forward(ActivationRelu().forward(output), layer2_wts, layer2_bia)
output3=layer3.forward(ActivationRelu().forward(output2), layer3_wts, layer3_bia)
output4=SoftmaxActivation().forward(output3)

print(output4)


print(layer1_wts)
print(layer2_wts)
print(layer3_wts)

print(layer1_bia)
print(layer2_bia)
print(layer3_bia)





        

        
            

        


