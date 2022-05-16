#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import numpy as np
import logging
import pyvista as pv
from pyDOE import lhs
from scipy import stats


LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
file_handler = logging.FileHandler(filename='test.log', mode='w')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# na straně serveru třída KLE 
class KLE(object):
    bone_type = ["Ilium R", "Ilium L", "Femur R"]  #datasets for bones
    #bone_type = ["Ilium R"]  #datasets for bones
    #bone_type =    ["Ilium R", "Ilium L", "Femur R", "Femur L", "Sacrum", "L5"]

    def __init__(self): 
        self.bone_type = KLE.bone_type
        self.bmd_mean_women = {}
        self.bmd_eigs_women = {}
        self.bmd_std_women = {}
        self.bmd_vecs_women = {}
        self.bmd_slopes_women = {}

        self.bmd_mean_men = {}
        self.bmd_eigs_men = {}
        self.bmd_std_men = {}
        self.bmd_vecs_men = {}
        self.bmd_slopes_men = {}

        #nahrat u klienta v panels aůe zde taky nechat
        #self.meshes = {}
        
        self._vecs = None
        self._eigs = None
        self._std = None
        self._mean = None
        self._slope = None
        self._sex = None
        self._thr = None
        self.data_loaded = False

        #ihned nacteme data v konstruktoru
        self.load_data()

    def load_data(self):
        logger.info('Loading data in KLE')
        for bone_type in self.bone_type:
            logger.info('Loading: ' + bone_type)
            self.bmd_mean_women[bone_type] = np.load(bone_type + '/bmd_mean_women.npy')
            self.bmd_eigs_women[bone_type] = np.diag(np.load(bone_type + '/bmd_eigs_women.npy'))
            self.bmd_std_women[bone_type] = np.load(bone_type + '/bmd_std_women.npy')
            self.bmd_vecs_women[bone_type] = np.load(bone_type + '/bmd_vecs_women.npy')
            self.bmd_slopes_women[bone_type] = np.load(bone_type + '/bmd_slopes_women.npy')
            self.bmd_mean_men[bone_type] = np.load(bone_type + '/bmd_mean_men.npy')
            self.bmd_eigs_men[bone_type] = np.diag(np.load(bone_type + '/bmd_eigs_men.npy'))
            self.bmd_std_men[bone_type] = np.load(bone_type + '/bmd_std_men.npy')
            self.bmd_vecs_men[bone_type] = np.load(bone_type + '/bmd_vecs_men.npy')
            self.bmd_slopes_men[bone_type] = np.load(bone_type + '/bmd_slopes_men.npy')
            #self.meshes[bone_type] = pv.read(bone_type + '/average_patient.xml') 
            #logger.info('Loading 11')    
        self.data_loaded = True
        logger.info('Data loaded in KLE')

    # sex =  hodnota z dict od clienta co nam psole
    def set_sex_and_bone_type(self, sex, bone_type):
        if sex=='woman':
            self._eigs = self.bmd_eigs_women[bone_type]
            self._vecs = self.bmd_vecs_women[bone_type]
            self._std = self.bmd_std_women[bone_type]
            self._mean = self.bmd_mean_women[bone_type]
            self._slope = self.bmd_slopes_women[bone_type]
            self._sex = 'woman'
        elif sex=='man':
            self._eigs = self.bmd_eigs_men[bone_type]
            self._vecs = self.bmd_vecs_men[bone_type]
            self._std = self.bmd_std_men[bone_type]
            self._mean = self.bmd_mean_men[bone_type]
            self._slope = self.bmd_slopes_men[bone_type]
            self._sex = 'man'
    
 
    def set_spectral_band(self, spb):   
        lt, ut = spb                                    
        self._thr = slice(lt, ut)

    def set_age(self,age):
        self.age = age

    def compute_realisation(self):
        logger.info('draw realisation')
        lhd = lhs(n=self._thr.stop - self._thr.start, samples=1)[0]
        xi = stats.norm().ppf(lhd)
        assert (self._thr.stop - self._thr.start) == len(xi)
        bmd_z= np.matmul(np.matmul(self._vecs[:, self._thr], 
                                   self._eigs[self._thr, self._thr]), 
                         np.diag(xi)).sum(1)
        return self._mean + self._slope * self.age + bmd_z * self._std


def create_socket():
    """Otevření socketu se specifikovaným typem spojení."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    return socket


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,        
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def start_server():
    """Spuštění serveru."""
    model = KLE()
    logging.info("Data loaded")
    socket = create_socket()
    while True:
        flags=0
        mpr= socket.recv_json(flags|zmq.SNDMORE)   
        model.set_sex_and_bone_type(mpr["sex"],mpr["bone_type"])
        model.set_spectral_band(mpr["spb"])
        model.set_age(mpr["age"])
        send_array(socket,model.compute_realisation())

start_server()



