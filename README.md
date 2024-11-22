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

- Miglior modello con Evaluation: -2.4402
- RMSE on VALIDATION : 2.9236
- SSIM on VALIDATION : 0.4835
- Evaluation Score on VALIDATION : -2.4402
- VALIDATION: RMSE=2.9236, SSIM=0.4835
- SUMMARY (Epoch 72): Train -> RMSE=1.3962, SSIM=0.7162 | Validation -> RMSE=2.9236, SSIM=0.4835               
- Epoch [73/100], Loss: 0.7741

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

Per raggiungere gli obiettivi della tua challenge in modo efficace, ti consiglio di adottare un approccio sperimentale strutturato in più fasi. Questo ti permetterà di trovare un equilibrio tra il miglioramento delle metriche e l’ottimizzazione delle risorse computazionali, considerando anche le peculiarità del dataset e le attuali soluzioni di stato dell’arte. Ecco un piano suggerito:

### Fase 1: **Convalida di Base con un Modello Pre-Addestrato (MiDaS/DPT)**
   - **Obiettivo**: Stabilire un punto di riferimento di performance (baseline) senza allenare il modello da zero.
   - **Approccio**: Utilizza MiDaS o DPT pre-addestrati per valutare i risultati su RMSE e SSIM con il dataset della challenge. 
   - **Esperimento**: Effettua una valutazione diretta con i dati di validazione e registra i risultati. Questo ti darà un’idea di quanto bene un modello pre-addestrato possa performare sul tuo dataset senza ulteriore addestramento.
   
### Fase 2: **Fine-Tuning dell’Encoder Pre-Addestrato con un Decoder Personalizzato**
   - **Obiettivo**: Integrare un decoder personalizzato che sia più specifico per il compito della challenge e addestrare il modello end-to-end.
   - **Approccio**:
     1. **Congela i primi layer** del modello MiDaS/DPT e allena soltanto le parti più alte del backbone insieme al nuovo decoder. Questo aiuta ad adattare le caratteristiche del modello al tuo dataset specifico senza compromettere le prestazioni iniziali.
     2. **Decoder Personalizzato**: Implementa un decoder che utilizzi tecniche di upsampling specifiche come convtranspose e skip connections (tipiche della U-Net) per migliorare la risoluzione spaziale della DepthMap.
   - **Esperimento**: Allena per un certo numero di epoche e confronta le metriche con i risultati della Fase 1.

### Fase 3: **Esplorazione di un Modello Interamente Personalizzato**
   - **Obiettivo**: Confrontare un modello encoder-decoder costruito da zero con i modelli pre-addestrati.
   - **Architetture Candidate**:
     - **U-Net con Encoder Personalizzato**: Questa rete può essere progettata per una mappatura dettagliata delle profondità, utilizzando un encoder adatto (come ResNet o EfficientNet).
     - **Alternative con EfficientNet o ConvNeXt**: Testare una variante con un encoder di EfficientNet o ConvNeXt, che offrono ottime prestazioni computazionali con una qualità delle feature alta.
   - **Esperimento**: Confronta RMSE e SSIM con le altre architetture pre-addestrate.

### Fase 4: **Valutazione e Confronto Finali**
   - **Obiettivo**: Documentare le prestazioni ottenute in ciascuna fase, spiegando il motivo delle scelte per ogni configurazione.
   - **Confronto**: Utilizza le metriche per confrontare tutte le architetture (MiDaS, fine-tuning su MiDaS, U-Net personalizzata) e seleziona l’approccio che meglio bilancia accuratezza e risorse.

Questo approccio sperimentale permette di documentare con rigore scientifico il motivo delle scelte architetturali e sperimentali, un elemento che sarà cruciale per la relazione finale della challenge.
