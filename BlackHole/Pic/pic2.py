
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

plt.imshow(Image.open("blackhole1080-t.png"))
plt.xticks([])
plt.yticks([])

cb = plt.colorbar(orientation='vertical',label='Time')
tick_locator = ticker.MaxNLocator(nbins=1)
cb.locator = tick_locator
cb.update_ticks()
plt.suptitle("Image of light travel time around a black hole."+"\n"+"r=22$R_s$"+" ; "+"$R_{disk}$=4.5$R_s$~15$R_s$")
plt.show()