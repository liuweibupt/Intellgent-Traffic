import tensorflow as tf

#拼接案例1
a=tf.random.normal([4,8])
b=tf.random.normal([4,8])
c=tf.stack([a,b],axis=0)
d=tf.reduce_mean(c,axis=0)
print(a)
print(b)
print(c)
print(c.shape)#(10, 35, 8)

print(d)

