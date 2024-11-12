# Depth Estimation Model

Questo progetto implementa un modello di stima della profondità utilizzando una rete encoder-decoder con un encoder basato su **ResNet34** pre-addestrato e un decoder che esegue operazioni di upsampling per generare mappe di profondità (DepthMaps) a partire da immagini RGB.

## Descrizione del Modello

- **Encoder**: ResNet34 pre-addestrato, con i primi 4 blocchi congelati.
- **Decoder**: Serie di livelli `ConvTranspose2d` con `BatchNorm2d` e `ReLU`, per la ricostruzione spaziale dell’immagine.

### Funzione di Perdita

La funzione di perdita combina:
- **L1 Loss** per garantire l'accuratezza numerica dei valori di profondità.
- **SSIM Loss** (Structural Similarity Index) per preservare la struttura dell'immagine.

## Dati di Addestramento

Le immagini di input sono immagini RGB in formato `.jpg`, mentre le ground truth di profondità sono fornite in formato `.npy`. 

## Risultati Migliori

Il modello ha ottenuto i seguenti risultati migliori durante l'allenamento:

- **Epoca migliore**: 83
- **RMSE**: 2.9697
- **SSIM**: 0.4728
- **Evaluation Score**: -2.4969

Questi risultati sono stati ottenuti dopo 100 epoche di allenamento, utilizzando pesi di `alpha = 1.0` per L1 Loss e `beta = 1.0` per SSIM Loss. 
**Nota**: i pesi della loss saranno prossimamente testati con `beta = 0.5` per verificare se ulteriori miglioramenti possono essere raggiunti.

## Utilizzo

### Requisiti

- Python >= 3.7
- PyTorch
- torchvision

### Addestramento del Modello

Per addestrare il modello, utilizzare lo script `main.py` e specificare i parametri necessari (directory dei dati, batch size, learning rate, numero di epoche, ecc.). 

```bash
python main.py --data_dir ./data --batch_size 32 --lr 0.001 --max_epochs 100
```

### Valutazione

La valutazione viene eseguita automaticamente al termine di ogni epoca, utilizzando il set di validazione. I migliori modelli sono salvati nella directory di checkpoint specificata.

### Salvataggio e Caricamento dei Modelli

Il modello è salvato automaticamente quando raggiunge la migliore metrica di valutazione. Può essere caricato per ulteriori test o per inferenza.

## Prossimi Passi

- Testare un valore ridotto di `beta = 0.5` per il peso della SSIM Loss.
- Introdurre la data augmentation per migliorare la generalizzazione del modello.
- Esplorare architetture alternative, come U-Net, per valutare eventuali miglioramenti nella stima della profondità.
