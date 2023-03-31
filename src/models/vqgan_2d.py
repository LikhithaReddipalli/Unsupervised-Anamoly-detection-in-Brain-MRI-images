import os
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from tqdm import tqdm
import torch
import wandb
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import utils as vutils
from src.models.modules.vqgan_architechture import Discriminator, weights_init
from src.models.modules.vqgan2d_model import vqgan_model_2d
from src.models.modules.lpips import LPIPS
from src.models.losses_vqgan import VQLPIPSWithDiscriminator
import torchvision.models as models
import torch.optim as optim
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, AUROC, AveragePrecision
from typing import Any, List
import warnings
import numbers
from mpl_toolkits.axes_grid1 import ImageGrid
from imageio import imwrite
from skimage.measure import regionprops, label
from torchvision.utils import save_image, make_grid
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve,accuracy_score, average_precision_score, normalized_mutual_info_score, precision_recall_fscore_support
import math
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import scipy
from sklearn.metrics import  roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support



class vqgan_2d_trainer(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vqgan_2d = vqgan_model_2d(cfg)
        self.loss = VQLPIPSWithDiscriminator(cfg)
        self.perceptual_loss = LPIPS().eval()
        self.save_hyperparameters()
        self.disc_factor = 1
        self.rec_loss_factor = 1.0
        self.disc_start = 2
        self.perceptual_loss_factor = 1.0
        self.opt_ae, self.opt_disc = self.configure_optimizers()


    def forward(self,x):
        x = self.vqgan_2d(x)
        return x

    def training_step(self, batch, batch_idx: int,  optimizer_idx):
        # process batch
        input = batch['vol'].unsqueeze(1)
        outputs = self(input)
        #print("outputs['x_recon']",outputs['x_recon'].shape)

        if optimizer_idx == 0:
            loss, loss_ = self.loss(outputs['vq_loss'], input, outputs['x_recon'], 0, self.global_step,
                                                  last_layer=self.get_last_layer())

            self.log('train/Loss_vq', loss_['aeloss'], prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/Loss_Recon', loss_['rec_loss'], prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/Loss_combines', loss_['combined'],prog_bar=True, on_step=False, on_epoch=True)



            # Plot reconstructed and original image in wandb while training
            if self.current_epoch % 50 == 0:

                orig_recon = torch.cat((input[0, 0, :, :],outputs['x_recon'][0, 0, :, :]),1)
                wandb.log({'recon_images': wandb.Image(orig_recon, caption=f'recon_{self.current_epoch}')})

            return {"loss": loss}


        if optimizer_idx == 1:
            discloss = self.loss(outputs['vq_loss'], input, outputs['x_recon'], 1, self.global_step,
                                            last_layer=self.get_last_layer())

            self.log('train/Loss_gan', discloss, prog_bar=True, on_step=False, on_epoch=True)


            return {"loss": discloss}


    def configure_optimizers(self):
        lr = 4.5e-6
        opt_ae = torch.optim.Adam(list(self.vqgan_2d.encoder.parameters()) +
                                  list(self.vqgan_2d.decoder.parameters()) +
                                  list(self.vqgan_2d.codebook.parameters()) +
                                  list(self.vqgan_2d.quant_conv.parameters()) +
                                  list(self.vqgan_2d.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def validation_step(self, batch, batch_idx: int):
        # process batch
        input = batch['vol'].unsqueeze(1)
        outputs = self(input)

        loss , loss_ = self.loss(outputs['vq_loss'], input, outputs['x_recon'], 0, self.global_step,
                                                  last_layer=self.get_last_layer())


        discloss = self.loss(outputs['vq_loss'],  input, outputs['x_recon'], 1, self.global_step,last_layer=self.get_last_layer())


        combined_loss = loss_['rec_loss'] + discloss

        self.log('val/Loss_vq', loss_['aeloss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/Loss_Recon', loss_['rec_loss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/Loss_comb', loss_['combined'], prog_bar=True, on_step=False, on_epoch=True) # changed
        self.log('val/Loss_gan', discloss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": combined_loss}


    def get_last_layer(self):

        last_layer = self.vqgan_2d.decoder.model[-1]
        last_layer_weight = last_layer.weight

        return last_layer_weight

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self, 'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):

        input = batch['vol']
        data_orig = batch['vol_orig']
        data_seg = batch['seg_orig']
        data_mask = batch['mask_orig']
        pad_params = batch['pad_params']
        ID = batch['ID']
        self.stage = batch['stage']

        if self.cfg.pad:
            size_x = 190
            size_y = 158
            num_slices = data_orig.shape[1]
            self.new_size = [num_slices, size_x, size_y]

        AnomalyScoreComb = []  # list to append combined loss of everyslice in a sample
        AnomalyScorevq = []   # vq loss
        AnomalyScoreReco = []  # recon loss

        final_volume = torch.zeros([input.size(1), input.size(2), input.size(3)])

        # New sample wise reconstruction and error calculation
        for i in range(input.size(1)):
            input_slice = input[:, i, :, :].view(1, 1, self.cfg.imageDim[1], self.cfg.imageDim[2])
            output_slice = self(input_slice)

            loss, loss_ = self.loss(output_slice['vq_loss'], input_slice, output_slice['x_recon'], 0, self.global_step,
                                    last_layer=self.get_last_layer())

            AnomalyScoreComb.append(loss_['combined'].item())
            AnomalyScorevq.append(loss_['aeloss'].item())
            AnomalyScoreReco.append(loss_['rec_loss'].item())

            # re assemble the reconstruction volume
            final_volume[i, :, :] = output_slice['x_recon']


        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb)  #combined loss of each sample
        AnomalyScorevq_vol = np.mean(AnomalyScorevq)  #vqloss of each sample
        AnomalyScoreReco_vol = np.mean(AnomalyScoreReco) #recon loss of each sample




        # Resize the images and remove the padding if desired
        if not self.cfg.resizedEvaluation:  # in case of full resolution evaluation
            final_volume = final_volume.unsqueeze(0).unsqueeze(0)
            final_volume = F.interpolate(final_volume, size=self.new_size, mode="trilinear",
                                         align_corners=True).squeeze()  # resize
            if self.cfg.get('undoPadding', False):  # undo the padding
                if pad_params[1] == 0:
                    final_volume = final_volume[pad_params[4]:-pad_params[5], pad_params[0]:,
                                   pad_params[2]:-pad_params[3]]  # because of dummy[0:-0] = []...
                if pad_params[3] == 0:
                    final_volume = final_volume[pad_params[4]:-pad_params[5], pad_params[0]:-pad_params[1],
                                   pad_params[2]:]  # because of dummy[0:-0] = []...
                if pad_params[5] == 0:
                    final_volume = final_volume[pad_params[4]:, pad_params[0]:-pad_params[1],
                                   pad_params[2]:-pad_params[3]]  # because of dummy[0:-0] = []...
                if pad_params[1] != 0 and pad_params[3] != 0 and pad_params[5] != 0:
                    final_volume = final_volume[pad_params[4]:-pad_params[5], pad_params[0]:-pad_params[1],
                                   pad_params[2]:-pad_params[3]]
        else:
            final_volume = final_volume.squeeze()
            if self.cfg.get('undoPadding', False):
                # rescale padding to the size of inputdata
                pad_params[0] = int(pad_params[0] * final_volume.squeeze().shape[1] / self.new_size[1])
                pad_params[1] = int(pad_params[1] * final_volume.squeeze().shape[1] / self.new_size[1])
                pad_params[2] = int(pad_params[2] * final_volume.squeeze().shape[2] / self.new_size[2])
                pad_params[3] = int(pad_params[3] * final_volume.squeeze().shape[2] / self.new_size[2])
                pad_params[4] = int(pad_params[4] * final_volume.squeeze().shape[0] / self.new_size[0])
                pad_params[5] = int(pad_params[5] * final_volume.squeeze().shape[0] / self.new_size[0])
                if pad_params[1] == 0:
                    final_volume = final_volume[:, pad_params[0]:,
                                   pad_params[2]:-pad_params[3]]  # because of dummy[0:-0] = []...
                if pad_params[3] == 0:
                    final_volume = final_volume[:, pad_params[0]:-pad_params[1],
                                   pad_params[2]:]  # because of dummy[0:-0] = []...
                if pad_params[5] == 0:
                    final_volume = final_volume[pad_params[4]:, pad_params[0]:-pad_params[1],
                                   pad_params[2]:-pad_params[3]]  # because of dummy[0:-0] = []...
                if pad_params[1] != 0 and pad_params[3] != 0 and pad_params[5] != 0:
                    final_volume = final_volume[pad_params[4]:-pad_params[5], pad_params[0]:-pad_params[1],
                                   pad_params[2]:-pad_params[3]]

                final_volume = final_volume.unsqueeze(0)
                final_volume = final_volume.unsqueeze(0)
                final_volume = F.interpolate(final_volume, size=(64, 64, 64), mode="bicubic",
                                             align_corners=True).squeeze()  # resize

        # move data to CPU
        data_orig = data_orig.cpu()
        final_volume = final_volume.cpu()

        # calculate the residual image
        diff_volume = torch.abs((data_orig - final_volume))

        # Calculate Reconstruction errors with respect to anomal/normal regions
        l1err = nn.functional.l1_loss(final_volume.squeeze(), data_orig.squeeze())
        l1err_anomal = nn.functional.l1_loss(final_volume.squeeze()[data_seg.squeeze() == 1], data_orig[data_seg == 1])
        l1err_healthy = nn.functional.l1_loss(final_volume.squeeze()[data_seg.squeeze() == 0], data_orig[data_seg == 0])
        l2err = nn.functional.mse_loss(final_volume.squeeze(), data_orig.squeeze())
        l2err_anomal = nn.functional.mse_loss(final_volume.squeeze()[data_seg.squeeze() == 1], data_orig[data_seg == 1])
        l2err_healthy = nn.functional.mse_loss(final_volume.squeeze()[data_seg.squeeze() == 0],
                                               data_orig[data_seg == 0])
        # store in eval dict
        self.eval_dict['l1recoErrorAll'].append(l1err.item())
        self.eval_dict['l1recoErrorUnhealthy'].append(l1err_anomal.item())
        self.eval_dict['l1recoErrorHealthy'].append(l1err_healthy.item())
        self.eval_dict['l2recoErrorAll'].append(l2err.item())
        self.eval_dict['l2recoErrorUnhealthy'].append(l2err_anomal.item())
        self.eval_dict['l2recoErrorHealthy'].append(l2err_healthy.item())

        # move data to CPU
        data_seg = data_seg.cpu()
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()

        # Erode the Brainmask
        if self.cfg['erodeBrainmask']:  # and state=='complete' :
            diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())

        # Filter the DifferenceImage
        if self.cfg['medianFiltering']:  # and state=='complete' :
            diff_volume = torch.from_numpy(apply_2d_median_filter(diff_volume.numpy().squeeze())).unsqueeze(
                0)  # bring back to tensor

        ### Compute Metrics per Volume(here slice in 2d)
        if self.cfg.evalSeg:

            # Pixel-Wise Segmentation Error Metrics based on Differenceimage(recon_image)
            AUC, _fpr, _tpr, _threshs = compute_roc(diff_volume.squeeze().flatten(),
                                                    np.array(data_seg.squeeze().flatten()).astype(bool))
            AUPRC, _precisions, _recalls, _threshs = compute_prc(diff_volume.squeeze().flatten(),
                                                                 np.array(data_seg.squeeze().flatten()).astype(
                                                                     bool))

            # gready search for threshold
            bestDice, bestThresh = find_best_val(np.array(diff_volume.squeeze()).flatten(),
                                                 # threshold search with a subset of EvaluationSet
                                                 np.array(data_seg.squeeze()).flatten().astype(bool),
                                                 val_range=(0, np.max(np.array(diff_volume))),
                                                 max_steps=10,
                                                 step=0,
                                                 max_val=0,
                                                 max_point=0)

            if 'test' in batch['stage']:
                bestThresh = self.threshold['total']
            if self.cfg["threshold"] == 'auto':
                diffs_thresholded = diff_volume > bestThresh
            else:  # never used
                diffs_thresholded = diff_volume > self.cfg["threshold"]

            # Connected Components
            diffs_thresholded = filter_2d_connected_components(
                np.squeeze(diffs_thresholded))  # this is only done for patient-wise evaluation atm

            # Calculate Dice Score with thresholded volumes
            diceScore = dice(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))

            # Classification Metrics
            TP, FP, TN, FN = confusion_matrix(np.array(diffs_thresholded.squeeze()),
                                              np.array(data_seg.squeeze().flatten()).astype(bool))
            TPR = tpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
            FPR = fpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
            self.eval_dict['lesionSizePerVol'].append(
                np.count_nonzero(np.array(data_seg.squeeze().flatten()).astype(bool)))
            self.eval_dict['DiceScorePerVol'].append(diceScore)
            self.eval_dict['BestDicePerVol'].append(bestDice)
            self.eval_dict['BestThresholdPerVol'].append(bestThresh)
            self.eval_dict['AUCPerVol'].append(AUC)
            self.eval_dict['AUPRCPerVol'].append(AUPRC)
            self.eval_dict['TPPerVol'].append(TP)
            self.eval_dict['FPPerVol'].append(FP)
            self.eval_dict['TNPerVol'].append(TN)
            self.eval_dict['FNPerVol'].append(FN)
            self.eval_dict['TPRPerVol'].append(TPR)
            self.eval_dict['FPRPerVol'].append(FPR)

            PrecRecF1PerVol = precision_recall_fscore_support(np.array(data_seg.squeeze().flatten()).astype(bool),
                                                              np.array(diffs_thresholded.squeeze()).flatten())
            self.eval_dict['AccuracyPerVol'].append(accuracy_score(np.array(data_seg.squeeze().flatten()).astype(bool),
                                                                   np.array(diffs_thresholded.squeeze()).flatten()))

            try:
                self.eval_dict['PrecisionPerVol'].append(PrecRecF1PerVol[0][1])

            except:
                pass
            try:
                self.eval_dict['RecallPerVol'].append(PrecRecF1PerVol[1][1])
            except:
                pass
            self.eval_dict['SpecificityPerVol'].append(TN / (TN + FP + 0.0000001))

            if batch_idx == 0:
                self.diffs_list = np.array(diff_volume.squeeze().flatten())
                self.seg_list = np.array(data_seg.squeeze().flatten()).astype(np.int8)
            else:
                self.diffs_list = np.append(self.diffs_list, np.array(diff_volume.squeeze().flatten()), axis=0)
                self.seg_list = np.append(self.seg_list, np.array(data_seg.squeeze().flatten()), axis=0).astype(np.int8)


            # compute slice-wise metrics
            for slice in range(data_seg.squeeze().shape[0]):
                if np.array(data_seg.squeeze()[slice].flatten()).astype(bool).any():
                    self.eval_dict['DiceScorePerSlice'].append(dice(np.array(diff_volume.squeeze()[slice] > bestThresh),
                                                                    np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))
                    PrecRecF1PerSlice = precision_recall_fscore_support(
                        np.array(data_seg.squeeze()[slice].flatten()).astype(bool),
                        np.array(diff_volume.squeeze()[slice] > bestThresh).flatten(), warn_for=tuple())
                    self.eval_dict['PrecisionPerSlice'].append(PrecRecF1PerSlice[0][1])
                    self.eval_dict['RecallPerSlice'].append(PrecRecF1PerSlice[1][1])
                    self.eval_dict['lesionSizePerSlice'].append(
                        np.count_nonzero(np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))

        # Reconstruction based Anomaly score for Slice-Wise evaluation
        AnomalyScoreReco = []  # Reconstruction based Anomaly score


        for slice in range(diff_volume.squeeze().shape[0]):
            score = diff_volume.squeeze()[slice][data_mask.squeeze()[slice]>0].mean()

            if score.isnan() : # if no brain exists in that slice
                AnomalyScoreReco.append(0.0)
            else:
                AnomalyScoreReco.append(score)

        # create slice-wise labels
        data_seg_downsampled = np.array(data_seg.squeeze())
        label = []  # store labels here
        for slice in range(data_seg_downsampled.shape[0]):  # iterate through volume
            if np.array(data_seg_downsampled[slice]).astype(bool).any():  # if there is an anomaly segmentation
                label.append(1)  # label = 1
            else:
                label.append(0)  # label = 0 if there is no Anomaly in the slice

        AUC, _fpr, _tpr, _threshs = compute_roc(np.array(AnomalyScoreReco), np.array(label))
        AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(AnomalyScoreReco), np.array(label))
        self.eval_dict['AUCAnomalyRecoPerSlice'].append(AUC)
        self.eval_dict['AUPRCAnomalyRecoPerSlice'].append(AUPRC)
        self.eval_dict['labelPerSlice'].extend(label)
        # store Slice-wise Anomalyscore (reconstruction based)
        self.eval_dict['AnomalyScoreRecoPerSlice'].extend(AnomalyScoreReco)
        # sample-Wise Anomalyscores
        self.eval_dict['AnomalyScorevqPerVol'].append(AnomalyScorevq_vol)
        self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
        if batch['Dataset'][0] == 'MN_IXI':  # for Healthy datasets the label is 0
            self.eval_dict['labelPerVol'].append(0)
        else:
            self.eval_dict['labelPerVol'].append(1)

    def on_test_end(self):

        # average over all test samples
        self.eval_dict['l1recoErrorAllMean'] = np.mean(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l1recoErrorAllStd'] = np.std(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l2recoErrorAllMean'] = np.mean(self.eval_dict['l2recoErrorAll'])
        self.eval_dict['l2recoErrorAllStd'] = np.std(self.eval_dict['l2recoErrorAll'])

        self.eval_dict['l1recoErrorHealthyMean'] = np.mean(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorHealthyStd'] = np.std(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorUnhealthyMean'] = np.mean(self.eval_dict['l1recoErrorUnhealthy'])
        self.eval_dict['l1recoErrorUnhealthyStd'] = np.std(self.eval_dict['l1recoErrorUnhealthy'])

        self.eval_dict['l2recoErrorHealthyMean'] = np.mean(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorHealthyStd'] = np.std(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorUnhealthyMean'] = np.mean(self.eval_dict['l2recoErrorUnhealthy'])
        self.eval_dict['l2recoErrorUnhealthyStd'] = np.std(self.eval_dict['l2recoErrorUnhealthy'])

        self.eval_dict['AUPRCPerVolMean'] = np.mean(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUPRCPerVolStd'] = np.std(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUCPerVolMean'] = np.mean(self.eval_dict['AUCPerVol'])
        self.eval_dict['AUCPerVolStd'] = np.std(self.eval_dict['AUCPerVol'])

        self.eval_dict['DicePerVolMean'] = np.mean(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['DicePerVolStd'] = np.std(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['BestDicePerVolMean'] = np.mean(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestDicePerVolStd'] = np.std(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestThresholdPerVolMean'] = np.mean(self.eval_dict['BestThresholdPerVol'])
        self.eval_dict['BestThresholdPerVolStd'] = np.std(self.eval_dict['BestThresholdPerVol'])

        self.eval_dict['TPPerVolMean'] = np.mean(self.eval_dict['TPPerVol'])
        self.eval_dict['TPPerVolStd'] = np.std(self.eval_dict['TPPerVol'])
        self.eval_dict['FPPerVolMean'] = np.mean(self.eval_dict['FPPerVol'])
        self.eval_dict['FPPerVolStd'] = np.std(self.eval_dict['FPPerVol'])
        self.eval_dict['TNPerVolMean'] = np.mean(self.eval_dict['TNPerVol'])
        self.eval_dict['TNPerVolStd'] = np.std(self.eval_dict['TNPerVol'])
        self.eval_dict['FNPerVolMean'] = np.mean(self.eval_dict['FNPerVol'])
        self.eval_dict['FNPerVolStd'] = np.std(self.eval_dict['FNPerVol'])
        self.eval_dict['TPRPerVolMean'] = np.mean(self.eval_dict['TPRPerVol'])
        self.eval_dict['TPRPerVolStd'] = np.std(self.eval_dict['TPRPerVol'])
        self.eval_dict['FPRPerVolMean'] = np.mean(self.eval_dict['FPRPerVol'])
        self.eval_dict['FPRPerVolStd'] = np.std(self.eval_dict['FPRPerVol'])
        self.eval_dict['PrecisionPerVolMean'] = np.mean(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['PrecisionPerVolStd'] = np.std(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['RecallPerVolMean'] = np.mean(self.eval_dict['RecallPerVol'])
        self.eval_dict['RecallPerVolStd'] = np.std(self.eval_dict['RecallPerVol'])
        self.eval_dict['PrecisionPerSliceMean'] = np.mean(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['PrecisionPerSliceStd'] = np.std(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['RecallPerSliceMean'] = np.mean(self.eval_dict['RecallPerSlice'])
        self.eval_dict['RecallPerSliceStd'] = np.std(self.eval_dict['RecallPerSlice'])
        self.eval_dict['AccuracyPerVolMean'] = np.mean(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['AccuracyPerVolStd'] = np.std(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['SpecificityPerVolMean'] = np.mean(self.eval_dict['SpecificityPerVol'])
        self.eval_dict['SpecificityPerVolStd'] = np.std(self.eval_dict['SpecificityPerVol'])

        # calculate the Metric across all samples
        if self.cfg.evalSeg:
            AUC, _fpr, _tpr, _threshs = compute_roc((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool))
            AUPRC, _precisions, _recalls, _threshs = compute_prc((self.diffs_list).flatten(),
                                                                 (self.seg_list).flatten().astype(bool))
            # gready search for the best Threshold based on validation set
            diceScore, bestThresh = find_best_val((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool),
                                                  val_range=(0, np.max((self.diffs_list))),
                                                  max_steps=10,
                                                  step=0,
                                                  max_val=0,
                                                  max_point=0)
            diffs_thresh_list = (self.diffs_list) > bestThresh

            if 'val' in self.stage:
                self.threshold['total'] = bestThresh
                # self.log('val/threshold', bestThresh )
            # further segmemntation Metrics
            TP, FP, TN, FN = confusion_matrix(np.array(diffs_thresh_list), (self.seg_list).flatten().astype(bool))
            TPR = tpr(np.array(diffs_thresh_list), (self.seg_list).flatten().astype(bool))
            FPR = fpr(np.array(diffs_thresh_list), (self.seg_list).flatten().astype(bool))
            self.eval_dict['SpecificityTotal'] = (TN / (TN + FP + 0.0000001))
            self.eval_dict['AccuracyTotal'] = accuracy_score(np.array(diffs_thresh_list),
                                                             (self.seg_list).flatten().astype(bool))
            self.eval_dict['DiceScoreTotal'] = diceScore

            self.eval_dict['AUCTotal'] = AUC
            self.eval_dict['AUPRCTotal'] = AUPRC
            self.eval_dict['ThresholdTotal'] = bestThresh

            # Save Metrics
            self.eval_dict['TPTotal'] = TP
            self.eval_dict['FPTotal'] = FP
            self.eval_dict['TNTotal'] = TN
            self.eval_dict['FNTotal'] = FN
            self.eval_dict['TPRTotal'] = TPR
            self.eval_dict['FPRTotal'] = FPR
            PrecRecF1Total = precision_recall_fscore_support(self.seg_list.flatten().astype(bool),
                                                             np.array(diffs_thresh_list))

            try:
                self.eval_dict['PrecisionTotal'] = PrecRecF1Total[0][1]
            except:
                pass
            try:
                self.eval_dict['RecallTotal'] = PrecRecF1Total[1][1]
            except:
                pass

            print('All Samples: ')
            print('\nTotal Dice Score: {} '.format(self.eval_dict['DiceScoreTotal']))
            print('Total AUC Score: {} '.format(self.eval_dict['AUCTotal']))
            print('Total AUPRC Score: {} '.format(self.eval_dict['AUPRCTotal']))
            print('\nPatient-Wise: ')
            print('\nAveraged Dice Score: {} +- {}'.format(self.eval_dict['DicePerVolMean'],
                                                           self.eval_dict['DicePerVolStd']))
            print(
                'Averaged AUC Score: {} +- {}'.format(self.eval_dict['AUCPerVolMean'], self.eval_dict['AUCPerVolStd']))
            print('Averaged AUPRC Score: {} +- {}'.format(self.eval_dict['AUPRCPerVolMean'],
                                                          self.eval_dict['AUPRCPerVolStd']))

            if 'test' in self.stage:
                del self.threshold

    def redFlagEvaluation(self, Sets, key_healthy, state):
        ### Calc AUC, AUPRC ###
        scores = [  # These are the scores where we combine the healthy and unhealthy sets
            'AnomalyScoreCombPerVol',
            'AnomalyScorevqPerVol',
            'AnomalyScoreRecoPerVol',
            'labelPerVol'
        ]
        eval_dict_redflag = {}  # output dict
        Set_healthy = Sets[state][key_healthy]
        Set_unhealthy = Sets[state]
        Set_unhealthy.pop(key_healthy)
        for set in Set_unhealthy:
            eval_dict_redflag[set] = {}
            for score in scores:
                Set_unhealthy[set][score].extend(Set_healthy[score])
            # Combined AnomalyScore  for each volume



            AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set_unhealthy[set]['AnomalyScoreCombPerVol']),
                                                    np.array(Set_unhealthy[set]['labelPerVol']))



            AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set_unhealthy[set]['AnomalyScoreCombPerVol']),
                                                                 np.array(Set_unhealthy[set]['labelPerVol']))


            eval_dict_redflag[set]['AUCperVolComb'] = AUC
            eval_dict_redflag[set]['AUPRCperVolComb'] = AUPRC
            # vqloss Term For each volume

            AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set_unhealthy[set]['AnomalyScorevqPerVol']),
                                                    np.array(Set_unhealthy[set]['labelPerVol']))


            AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set_unhealthy[set]['AnomalyScorevqPerVol']),
                                                                 np.array(Set_unhealthy[set]['labelPerVol']))


            eval_dict_redflag[set]['AUCperVolvq'] = AUC
            eval_dict_redflag[set]['AUPRCperVolvq'] = AUPRC
            # Reconstruction Term for each volume



            AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set_unhealthy[set]['AnomalyScoreRecoPerVol']),
                                                        np.array(Set_unhealthy[set]['labelPerVol']))



            AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set_unhealthy[set]['AnomalyScoreRecoPerVol']),
                                                                 np.array(Set_unhealthy[set]['labelPerVol']))

            eval_dict_redflag[set]['AUCperVolReco'] = AUC
            eval_dict_redflag[set]['AUPRCperVolReco'] = AUPRC

            # Slicewise Reconstruction Anomalyscore
            AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set_unhealthy[set]['AnomalyScoreRecoPerSlice']),
                                                    np.array(Set_unhealthy[set]['labelPerSlice']))
            AUPRC, _precisions, _recalls, _threshs = compute_prc(
                np.array(Set_unhealthy[set]['AnomalyScoreRecoPerSlice']), np.array(Set_unhealthy[set]['labelPerSlice']))
            eval_dict_redflag[set]['AUCperSliceReco'] = AUC
            eval_dict_redflag[set]['AUPRCperSliceReco'] = AUPRC

        return eval_dict_redflag

def get_eval_dictionary():
    _eval = {
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'Segmentation': [],
        'reconstructionTimes': [],
        'latentSpace': [],
        'Age': [],
        'AgeGroup': [],
        'l1reconstructionErrors': [],
        'l1recoErrorAll': [],
        'l1recoErrorUnhealthy': [],
        'l1recoErrorHealthy': [],
        'l2recoErrorAll': [],
        'l2recoErrorUnhealthy': [],
        'l2recoErrorHealthy': [],
        'l1reconstructionErrorMean': 0.0,
        'l1reconstructionErrorStd': 0.0,
        'l2reconstructionErrors': [],
        'l2reconstructionErrorMean': 0.0,
        'l2reconstructionErrorStd': 0.0,

        'TPPerVol': [],
        'FPPerVol': [],
        'FNPerVol': [],
        'TNPerVol': [],
        'TPRPerVol': [],
        'FPRPerVol': [],
        'TPTotal': [],
        'FPTotal': [],
        'FNTotal': [],
        'TNTotal': [],
        'TPRTotal': [],
        'FPRTotal': [],

        'PrecisionPerVol': [],
        'RecallPerVol': [],
        'PrecisionPerSlice': [],
        'RecallPerSlice': [],
        'lesionSizePerSlice': [],
        'lesionSizePerVol': [],
        'Dice': [],
        'DiceScorePerSlice': [],
        'DiceScorePerVol': [],
        'BestDicePerVol': [],
        'BestThresholdPerVol': [],
        'AUCPerVol': [],
        'AUPRCPerVol': [],
        'SpecificityPerVol': [],
        'AccuracyPerVol': [],

        'AUCAnomalyCombPerSlice': [],  # PerVol!!! + Confusionmatrix.
        'AUPRCAnomalyCombPerSlice': [],
        'AnomalyScoreCombPerSlice': [],


        'AUCAnomalyRecoPerSlice': [],
        'AUPRCAnomalyRecoPerSlice': [],
        'AnomalyScoreRecoPerSlice': [],


        'labelPerSlice': [],
        'labelPerVol': [],
        'AnomalyScoreCombPerVol': [],
        'AnomalyScoreCombMeanPerVol': [],
        'AnomalyScorevqPerVol': [],
        'AnomalyScorevqMeanPerVol': [],
        'AnomalyScoreRecoPerVol': [],
        'AnomalyScoreAgePerVol': [],
        'AnomalyScoreRecoMeanPerVol': []
    }
    return _eval


def apply_brainmask(x, brainmask, erode, iterations):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel,
                                                            iterations=iterations)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))


def apply_brainmask_volume(vol, mask_vol, erode=True, iterations=10):
    for s in range(vol.squeeze().shape[0]):
        slice = vol.squeeze()[:, :,s]
        mask_slice = mask_vol.squeeze()[:, :,s]
        eroded_vol_slice = apply_brainmask(slice, mask_slice, erode=True, iterations=vol.squeeze().shape[1] // 16)
        vol.squeeze()[:, :,s] = eroded_vol_slice
    return vol


def apply_2d_median_filter(volume, kernelsize=9):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume


def squash_intensities(img):
    # logistic function intended to squash reconstruction errors from [0;0.2] to [0;1] (just an example)
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)


def apply_colormap(img, colormap_handle):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = Image.fromarray(np.uint8(colormap_handle(img) * 255))
    return img


def add_colorbar(img):
    for i in range(img.squeeze().shape[0]):
        img[i] = float(i) / img.squeeze().shape[0]

    return img


def filter_2d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume


# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  # x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5
    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    # print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    # print(str(np.mean(x>q_top)) + str(np.mean(y)))
    # val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    # val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)


def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score


def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds




def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


def confusion_matrix(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    return tp, fp, tn, fn


def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tn)


