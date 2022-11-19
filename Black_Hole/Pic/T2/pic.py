import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# plt.style.use('seaborn-white')
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_title(r"$\theta$="+str(i*10)+"°")
    ax.imshow(Image.open("blackhole-"+ str(i*10) +".png"))
fig.suptitle("Images of a black hole from different angles."+"\n"+"r=22$R_s$"+" ; "+"$R_{disk}$=4.5$R_s$~15$R_s$")
plt.show()