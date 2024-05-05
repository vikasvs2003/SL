import numpy as np

def linear_thershold(dot,T):
    if dot >= T:
        return 1
    else :
        return 0
    
input_table = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

weights = np.array([1,-1])
T=1

dot_product_sum = input_table@weights

print(f'input table :\n{input_table}')
print(f'dot_products sum : \n{dot_product_sum}')

for i in range(0,4):
    activation = linear_thershold(dot_product_sum[i],T)
    print(f'activation :\n{activation}')