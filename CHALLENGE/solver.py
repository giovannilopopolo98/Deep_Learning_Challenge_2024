import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DepthDataset
from torchvision import transforms, models
from utils import visualize_img, ssim
from model import DepthEstimationModel
from torch.optim.lr_scheduler import CyclicLR

class Solver():

    def __init__(self, args):
        # Prepara il dataset
        self.args = args

        # Definizione delle trasformazioni per il dataset di training
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),                      # Ridimensiona le immagini a 224x224
            transforms.CenterCrop((224, 224)),                  # Ridimensiona le immagini alla risoluzione desiderata
        ])

        # Trasformazioni per il dataset di validazione
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),                       
            transforms.CenterCrop((224, 224)),                    
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inizializzazione del modello DepthEstimationModel
        self.net = DepthEstimationModel().to(self.device)  # Assicurati che venga inizializzato indipendentemente da `is_train`

        # Creazione dei DataLoader per il training e la validazione
        if self.args.is_train:

            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=train_transforms
                                           )
            
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir,
                                         transform=val_transforms
                                         )

            # Creazione dei DataLoader per il training e la validazione
            self.train_loader = DataLoader(dataset = self.train_data,
                                           batch_size=args.batch_size,
                                           num_workers=4,  
                                           shuffle=True, 
                                           drop_last=True)
            
            self.val_loader = DataLoader(dataset = self.val_data,
                                            batch_size=args.batch_size,
                                            num_workers=4, 
                                            shuffle=False, 
                                            drop_last=False)
            
            
            # Inizializzazione dell'ottimizzatore
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
            self.scheduler = CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.005, step_size_up=1000, step_size_down=2000, mode='triangular2')

            # Inizializzazione della miglior valutazione 
            self.best_evaluation_score = float('-inf')

            # Crea la directory per checkpoint se non esiste
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

        else:
            test_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
            ])

            self.test_set = DepthDataset(train=DepthDataset.TEST,       
                                         data_dir=self.args.data_dir,
                                         transform=test_transforms
                                         )  
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))  # weights_only=True per caricare solo i pesi, e non l'architettura (utile se l'architettura è la stessa tra training e test)
    

    def fit(self):

        for epoch in range(self.args.max_epochs):
            self.net.train() # Setta il modello in modalità di training
            epoch_loss = 0.0 # Inizializza la loss per l'epoca

            for i, (images, depth) in enumerate(self.train_loader):
                
                # Assicura che le immagini e la depth siano inviate al dispositivo (GPU o CPU)
                images = images.to(self.device)

                # Assicura che la depth sia inviata al dispositivo (GPU o CPU)
                depth = depth.to(self.device)

                # Forward pass
                output = self.net(images) # Calcola l'output del modello

                # Calcolo della loss combinata
                ssim_loss = 1 - ssim(output, depth) # Calcola la loss SSIM per la similarità strutturale 
                rmse_loss =  torch.sqrt(F.mse_loss(output, depth)) # Calcola la loss RMSE per l'errore quadratico medio
                
                # Pesi delle diverse loss
                alpha = 1.0
                beta = 1.0

                loss = alpha*rmse_loss + beta*ssim_loss # Calcola la loss totale come combinazione pesata delle loss

                # Backward pass
                self.optimizer.zero_grad() # Azzera i gradienti
                loss.backward()            # Calcola i gradienti

                self.optimizer.step()      # Aggiorna i pesi
                self.scheduler.step()     # Aggiorna il learning rate

                epoch_loss += loss.item()  # Aggiorna la loss dell'epoca

            # Stampa della Loss media per epoca 
            print(f"Epoch [{epoch + 1}/{self.args.max_epochs}], Loss: {epoch_loss / len(self.train_loader):.4f}")

            # Valuta sul set di training e validazione
            if (epoch + 1) % self.args.evaluate_every == 0:
                print(f"\n--- Evaluation for Epoch [{epoch + 1}/{self.args.max_epochs}] ---")
                
                print("Evaluating on TRAIN set...")
                self.evaluate(DepthDataset.TRAIN, epoch+1)

                print("Evaluating on VALIDATION set...")
                self.evaluate(DepthDataset.VAL, epoch+1)

                print(f"\n--- End of Evaluation for Epoch {epoch + 1} ---\n")

    def evaluate(self, set, epoch):

        args = self.args

        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
            save_model = False   # Non salvare il modello quando sta valutando il set di training
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
            save_model = True    # Salva il modello quando sta valutando il set di validazione
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
                    visualize_img(images[0].cpu().clamp(0, 1),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)
                    
        # Calcola la media delle metriche di valutazione
        avg_rmse = rmse_acc / len(loader)
        avg_ssim = ssim_acc / len(loader)
        
        # Calcola uno score combinato: minimizza RMSE e massimizza SSIM
        # L'obiettivo è massimizzare tale evaluation_score, quindi usiamo -RMSE e +SSIM
        evaluation_score = -avg_rmse + avg_ssim # Valore da massimizzare

        # Salva il modello se il punteggio di valutazione combinato del Validation è il migliore finora
        if save_model and evaluation_score > self.best_evaluation_score:
            self.best_evaluation_score = evaluation_score
            print(f"Salvataggio del miglior modello con Evaluation: {self.best_evaluation_score:.4f}")
            self.save(self.args.ckpt_dir, self.args.ckpt_name, epoch)

        # Stampa i risultati della valutazione
        print(f"RMSE on {suffix} : {avg_rmse:.4f}")
        print(f"SSIM on {suffix} : {avg_ssim:.4f}")
        print(f"Evaluation Score on {suffix} : {evaluation_score:.4f}")

        return avg_rmse, avg_ssim

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

    def test(self):

        loader = DataLoader(self.test_set, 
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu().clamp(0, 1),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="VALIDATION")
                    
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))
