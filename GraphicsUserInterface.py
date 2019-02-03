
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy

class GraphicsUserInterface:

    def drawArray(self, pixels : list, width : int, height : int):
        plt.imshow(pixels)
        plt.show()

        
    def drawArrayCompare(self, title1 : str, title2 : str, 
                         pixels1 : list, pixels2 : list, width : int, height : int):
        fig, axarr = plt.subplots(1,2)
        axarr[0].imshow(numpy.reshape(pixels1, [width, height]), cmap='gray')
        axarr[1].imshow(numpy.reshape(pixels2, [width, height]), cmap='gray')
        axarr[0].set_title(title1)
        axarr[1].set_title(title2)
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