import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DepthDataset
from torchvision import transforms, models
from utils import visualize_img, ssim
from model import DepthEstimationModel


class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inizializzazione del modello DepthEstimationModel
        self.net = DepthEstimationModel().to(self.device)  # Assicurati che venga inizializzato indipendentemente da `is_train`

        # Inizializzazione dell'ottimizzatore
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)

        # Inizializzazione della miglior valutazione 
        self.best_evaluation_score = float('-inf')

        # Creazione dei DataLoader per il training e la validazione
        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir)
            
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir)

            # Creazione dei DataLoader per il training e la validazione
            self.train_loader = DataLoader(self.train_data,
                                           batch_size=args.batch_size,
                                           num_workers=4,  
                                           shuffle=True, drop_last=True)
            
            self.val_loader = DataLoader(self.val_data,
                                            batch_size=args.batch_size,
                                            num_workers=4, 
                                            shuffle=False, drop_last=False)

            # Crea la directory per checkpoint se non esiste
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

        else:
            self.test_set = DepthDataset(train=DepthDataset.TEST, data_dir=self.args.data_dir)
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))
    

    def fit(self):
        for epoch in range(self.args.max_epochs):
            self.net.train() # Setta il modello in modalità di training
            epoch_loss = 0.0 # Inizializza la loss per l'epoca

            for i, (images, depth) in enumerate(self.train_loader):
                
                # Assicurati che le immagini e la depth siano inviate al dispositivo (GPU o CPU)
                images = images.to(self.device)

                # Assicurati che la depth sia inviata al dispositivo (GPU o CPU)
                depth = depth.to(self.device)

                # Forward pass
                output = self.net(images) # Calcola l'output del modello

                # Calcolo della loss combinata
                l1_loss = F.l1_loss(output, depth)  # Calcola la loss L1 per l'accuratezza numerica della DepthMap
                ssim_loss = 1 - ssim(output, depth) # Calcola la loss SSIM per la similarità strutturale 
                # hub_loss = F.huber_loss(output, depth)

                # Pesi delle diverse loss
                alpha = 1.0
                beta = 1.0
                # gamma = 0.0

                loss = alpha*l1_loss + beta*ssim_loss # Calcola la loss totale 

                # Backward pass
                self.optimizer.zero_grad() # Azzera i gradienti
                loss.backward()            # Calcola i gradienti
                self.optimizer.step()      # Aggiorna i pesi

                epoch_loss += loss.item()  # Aggiorna la loss dell'epoca

                # Visualizzazione periodica dell'imamgine di input, della DepthMap reale e di quella predetta
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TRAIN")
                    
            # Stampa della perdita media per epoca 
            print(f"Epoch [{epoch + 1}/{self.args.max_epochs}], Loss: {epoch_loss / len(self.train_loader):.4f}")

            # Salvataggio del modello ogni 'evaluate_every' epoche
            if (epoch + 1) % self.args.evaluate_every == 0:

                # Valutazione del modello
                self.evaluate(DepthDataset.VAL, epoch)


    def evaluate(self, set, epoch):

        args = self.args

        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0

        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):

                # Assicurati che le immagini e la depth siano inviate al dispositivo (GPU o CPU)
                images = images.to(self.device)
                depth = depth.to(self.device)

                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
                    
        # Calcola la media delle metriche di valutazione
        avg_rsme = rmse_acc / len(loader)
        avg_ssim = ssim_acc / len(loader)
        
        # Calcola uno score combinato: minimizza RMSE e massimizza SSIM
        # L'obiettivo è massimizzare tale evaluation_score, quindi usiamo -RMSE e +SSIM
        evaluation_score = -avg_rsme + avg_ssim # Valore da massimizzare

        # Salva il modello se il punteggio di valutazione combinato è il migliore finora
        if evaluation_score > self.best_evaluation_score:
            self.best_evaluation_score = evaluation_score
            print(f"Salvataggio del miglior modello con Evaluation: {self.best_evaluation_score:.4f}")
            self.save(self.args.ckpt_dir, self.args.ckpt_name, epoch)

        # Stampa i risultati della valutazione
        print(f"RMSE on VALIDATION : {avg_rsme:.4f}")
        print(f"SSIM on VALIDATION : {avg_ssim:.4f}") 
        print(f"Evaluation Score on VALIDATION : {evaluation_score:.4f}")

        return avg_rsme, avg_ssim

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))