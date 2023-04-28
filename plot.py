import matplotlib.pyplot as plt


train_loss_file = 'output/losses/train loss.txt'
eval_loss_file = 'output/losses/eval loss.txt'

with open(train_loss_file, 'r') as f:
    train_loss = [float(loss.rstrip()) for loss in f.readlines()]
with open(eval_loss_file, 'r') as f:
    eval_loss = [float(loss.rstrip()) for loss in f.readlines()]
plt.title('LOSS')
plt.xlabel('iter')
plt.ylabel('loss')
plt.plot(train_loss, label='train loss', color='b')
plt.plot(eval_loss, label='eval loss', color='r')
plt.legend(loc='best')
plt.show()