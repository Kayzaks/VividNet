
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy

class GraphicsUserInterface:
        
    def drawAll(self, pixels1 : list, pixels2 : list, width1 : int, height1 : int, width2 : int, height2 : int, 
                         semantics : list = [], texts : list = []):
        fig, axarr = plt.subplots(1,3)
        axarr[0].imshow(numpy.reshape(pixels1, [height1, width1, 3]))
        axarr[1].imshow(numpy.reshape(pixels1, [height1, width1, 3]))
        axarr[2].imshow(numpy.reshape(pixels2, [height2, width2, 3]))
        axarr[0].set_title("Original")
        axarr[1].set_title("Detected Semantics")
        axarr[2].set_title("'Mind's eye'")

        for semantic in semantics:
            axarr[1].add_patch(semantic)

        for text in texts:
            axarr[1].text(text[0], text[1], text[2], color = 'y', fontsize=8)

        plt.show()
        

    def createDrawBoard(self, lambdaRunCallback, lambdaCloseCallback, intDimX, intDimY):

        def mouseMove(event):
            if event.xdata is None or event.ydata is None or event.button is None:
                return

            listPixels[int(event.ydata)][int(event.xdata)] = 1
            img.set_data(listPixels)
            fig.canvas.draw()

        def runButton(event):
            lambdaRunCallback(listPixels)
            plt.close()
            lambdaCloseCallback()

        listPixels = numpy.zeros((intDimX, intDimY))
        listPixels[0][0] = 1

        fig, ax = plt.subplots(1,1)

        img = ax.imshow(listPixels, cmap='gray')

        listPixels[0][0] = 0
        img.set_data(listPixels)

        fig.canvas.mpl_connect('motion_notify_event', mouseMove)
        axbtn = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axbtn, 'Run')
        bnext.on_clicked(runButton)

        plt.show()