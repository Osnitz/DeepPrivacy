import os
import subprocess

if __name__ == '__main__':
    # XM2VTS
    #dataset_path = '../../datasets/xm2vts_aligned'
    #dataset_save = '../../datasets/xm2vts_deidentified'
    # RAFD
    #dataset_path = '../../datasets/rafd_aligned'
    #dataset_save = '../../datasets/rafd_DeepPrivacy_deidentified'
    dataset_path = '/media/blaz/Storage/datasets/face datasets/emotion/RaFD2 - Radboud Faces Database/RafDDownload-90_45_135_aligned'
    dataset_save = '../../datasets/rafd45_DeepPrivacy'        
    # CelebA-HQ (test)
    #dataset_path = '../../datasets/celeba-test_aligned'
    #dataset_save = '../../datasets/celeba-test_DeepPrivacy_deidentified'
    # AffectNet (val)
    #dataset_path = '../../datasets/AffectNet_aligned'
    #dataset_save = '../../datasets/AffectNet_DeepPrivacy_deidentified'
    dataset_path = '/home/matthieup/deid-toolkit/root_dir/datasets/aligned/fri'      
    dataset_save = '/home/matthieup/deid_techniques_and_evaluation/techniques/DeepPrivacy/Test_results'

    dataset_filetype = 'jpg'
    dataset_newtype = 'jpg'

    img_names = [i for i in os.listdir(dataset_path) if dataset_filetype in i] # change ppm into jpg
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    save_paths = [os.path.join(dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]

    for img_path, save_path in zip(img_paths, save_paths):
        p = subprocess.Popen(['python', 'anonymize.py', '-s', img_path, '-t', save_path],
        bufsize=2048, stdin=subprocess.PIPE)
        #p.stdin.write('e')
        p.wait()
        if p.returncode == 0:
            print("Image: ", img_path, " processed OK.")
        else:
            print("Image: ", img_path, " FAILED.")

    
