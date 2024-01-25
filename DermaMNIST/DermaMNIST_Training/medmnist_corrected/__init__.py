from medmnist_corrected.info import __version__, HOMEPAGE, INFO
try:
    from medmnist_corrected.dataset import (PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST,
                                  BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
                                  OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D,
                                  DermaMNIST_Corrected_28, DermaMNIST_Corrected_224,
                                  DermaMNIST_Extended_28, DermaMNIST_Extended_224)
    from medmnist_corrected.evaluator import Evaluator
except:
    print("Please install the required packages first. " +
          "Use `pip install -r requirements.txt`.")
