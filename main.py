import sys
from PyQt5.QtWidgets import QApplication
from interface import IntelligentLensSystem

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = IntelligentLensSystem()
    mainWin.show()
    sys.exit(app.exec_())
    #########EW####
