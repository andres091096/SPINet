import torch
import numpy as np

from skimage import measure

def fwht(a) -> None:
    """In-place Inverse Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h] 
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def cake_cutting(Hadamard):
  n = Hadamard.shape[0]
  p = int(np.sqrt(n))
  normal_index = np.arange(n)
  cc_order = []
  for i in range(n): 
    _,num = measure.label(Hadamard[i].reshape(p,p),return_num=True, connectivity=1)
    cc_order.append(num)

  row_index = [x for _,x in sorted(zip(cc_order,normal_index))]
  return row_index

def TV_order(Hadamard):
  n = Hadamard.shape[0]
  p = int(np.sqrt(n))
  normal_index = np.arange(n)
  tv_order = []
  for i in range(n):
    U = Hadamard[i].reshape(p,p)
    x_grad = np.vstack(((np.diff(U, axis=0)), (U[0,:] - U[-1,:]))).T
    y_grad = np.vstack(((np.diff(U, axis=1)).T, (U[:,0] - U[:,-1])))
    tv_order.append( np.sum( np.sqrt( (x_grad**2 + y_grad**2) ) ) )  

  row_index = [x for _,x in sorted(zip(tv_order,normal_index))]
  return row_index

def gen_sensing_matrix(features, measures, colum_perm, row_perm):
  I_matrix = np.identity(features)
  temp = np.zeros((features,features))

  A_matrix    = np.zeros((measures,features))
  Temp_Matrix = np.zeros((measures,features))

  for i in range(features):
    temp[i,:] = fwht(I_matrix[:,i])

  if row_perm == 'normal':
    row_index = np.arange(features)
  elif row_perm == 'sequential':
    f=lambda features:[int(bin(i+2**features)[:1:-1],2)//2 for i in range(2**features)] 
    row_index = f(int(np.log2(features)))
  elif row_perm == 'random':
    row_index = np.random.permutation(features)
  elif row_perm == 'high_frequency':
    f=lambda features:[int(bin(i+2**features)[:1:-1],2)//2 for i in range(2**features)] 
    row_index = np.flipud(f(int(np.log2(features))))
  elif row_perm == 'tv_order':
    row_index = TV_order(temp)
  elif row_perm == 'cake_cutting':
    row_index = cake_cutting(temp)

  if colum_perm == 'normal':
    col_index = np.arange(features)
  elif colum_perm == 'random':
    col_index = np.random.permutation(features)
  #Apply row permutations
  for i in range(measures):
    Temp_Matrix[i,:] = temp[row_index[i],:]
  #Apply column permutations
  for i in range(features):
    A_matrix[:,i] = Temp_Matrix[:,col_index[i]]
  
  return A_matrix, row_index, col_index

class AWGN(object):
    def __init__(self, SNR_dB):
      self.SNR  = pow(10,(SNR_dB/10)); #SNR to linear scale
    def __call__(self, x):
      L = len(x)
      Esym = np.sum(np.abs(pow(x,2)))/(L); #Calculate actual symbol energy
      k    = Esym/self.SNR; #Find the noise spectral density
      noiseSigma = np.sqrt(k); #Standard deviation for AWGN Noise 
      noise = noiseSigma*np.random.randn(L); #computed noise
      return x + noise

class SensingDiferential(object):
    """Transform a numpy array with a sensing matrix computed offline.
    Given sensing_matrix, will flatten the numpy array and compute the dot
    product with the sensing_matrix matrix and then reshaping the tensor to its
    original shape.
    Args:
        sensing_matrix (Tensor): tensor [D x D], D = C x H x W
    """

    def __init__(self, transformation_matrix):
        self.transformation_matrix = transformation_matrix

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image array of size (H, W) to be whitened.
        Returns:
            array: Differential Measure.
        """
        #if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
        #    raise ValueError("tensor and transformation matrix have incompatible shape." +
        #                     "[{} x {} x {}] != ".format(*tensor.size()) +
        #                     "{}".format(self.transformation_matrix.size(0)))
   
        m,n = self.transformation_matrix.shape
        measure = np.zeros((n))
        measure[:m] = np.dot(self.transformation_matrix,np.array(img).flatten())
        return measure

    def __repr__(self):
        format_string = self.__class__.__name__ + '(transformation_matrix='
        format_string += (str(self.transformation_matrix.tolist()) + ')')
        return format_string

class InverseFWHT(object):
    """Inverse  Walsh Hadamard Transform.
    Args:
        row_permutations (array): numpy array [N], N = H x W
    """

    def __init__(self, row_index, col_index):
        self.row_index = row_index
        self.col_index = col_index

    def __call__(self, b):
        """
        Args:
            img (PIL Image): PIL Image array of size (H, W) to be whitened.
        Returns:
            array: Differential Measure.
        """
        #if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
        #    raise ValueError("tensor and transformation matrix have incompatible shape." +
        #                     "[{} x {} x {}] != ".format(*tensor.size()) +
        #                     "{}".format(self.transformation_matrix.size(0)))
        
        img_size = len(self.row_index)
        reconst = np.zeros((img_size))
        reconst[self.row_index] = b
   
        h = 1
        while h < len(reconst):
          for i in range(0, len(reconst), h * 2):
            for j in range(i, i + h):
                x = reconst[j]
                y = reconst[j + h] 
                reconst[j] = x + y
                reconst[j + h] = x - y
          h *= 2
        
        imin = reconst.min()
        imax = reconst.max()

        a = (255) / (imax - imin)
        b = 255 - a * imax
        new_img = (a * reconst + b).astype(np.uint8)
        return new_img.reshape(int(np.sqrt(img_size)),int(np.sqrt(img_size)))

    def __repr__(self):
      format_string = self.__class__.__name__ + '(row_index='
      format_string += (str(self.row_index.tolist()) + ')')
      return format_string

class TestDatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, data, targets, transforms):
        super(TestDatasetFromFolder,self).__init__()
        self.data = data
        self.targets = targets
        self.transforms = transforms

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        
        return (self.transforms(img), self.transforms(target))

    def __len__(self):
        return len(self.data)