# Plan d'Expérience : Transfert OOD Inter-Cohorte en Régime Few-Label (Analogue Mueller / PI-VAE)

Ce plan vise à reproduire la méthodologie de l'expérience de transfert hors-distribution (OOD) en régime de données rares (low-label) appliquée à l'EEG, en utilisant un modèle de représentation JEPA et le VAE Spectral comme générateur physiquement informé.

> [!IMPORTANT]
> **Modèle JEPA non existant :** Actuellement, aucun checkpoint JEPA pré-entraîné n'existe sur le cluster. La première étape obligatoire sera donc de coder l'encodeur et l'objectif SSL, puis de lancer le pré-entraînement massif non-supervisé.

## 1. Pré-entraînement du modèle de base (JEPA SSL)
**Objectif :** Obtenir un extracteur de caractéristiques robuste et invariant.
- **Fichier à modifier :** `examples/eeg/main.py`
- **Architecture (`build_encoder`) :** Un réseau Conv1D profond (ResNet 1D) prenant des fenêtres `[B, 19, T]`, avec une projection latente globale `[B, D]`.
- **Objectif SSL (`build_ssl`) :** Implémentation de la fonction de perte VICReg (Invariance, Variance, Covariance) sur les deux vues natives du dataset, ou un JEPA prédictif temporel si préféré.
- **Exécution :** Lancement d'un job Slurm sur `defq` pour pré-entraîner l'encodeur sur les fenêtres non-labellisées de la cohorte d'entraînement.

## 2. Benchmark de Transfert OOD Low-Label (Patient-Disjoint)
**Objectif :** Évaluer la capacité de transfert de l'encodeur JEPA gelé avec différentes stratégies d'augmentation de données sur des fractions de données labellisées (1%, 5%, 25%, 100%).
- **Script :** Création de `scripts/benchmark_jepa_ood.py`
- **Validation :** L'évaluation se fera strictement au niveau du patient sur la cohorte de test (patient-disjoint, analogue à "test externe Cohorte 1").

### Stratégies Comparées :
1. **JEPA sans augmentation :** Entraînement d'une sonde linéaire (ou MLP) directement sur les embeddings JEPA des données réelles.
2. **JEPA + Augmentations classiques :** Les signaux réels subissent du bruit blanc, scale jitter, channel dropout, passent dans le JEPA gelé, puis entraînent le classifieur.
3. **JEPA + VAE Spectral :** Les signaux réels sont reconstruits/générés par le VAE pré-entraîné (`vae_checkpoint_final.pt`), passés dans le JEPA, puis utilisés pour entraîner le classifieur.
4. **JEPA + VAE Spectral downstream :** *(Clarification requise ci-dessous)*

> [!WARNING]
> **Questions Ouvertes**
> 
> Dans votre formulation pour les stratégies de test, vous citez :
> 1. JEPA + VAE spectral
> 2. JEPA + VAE spectral downstream
> 
> Pourriez-vous clarifier la différence exacte souhaitée entre ces deux approches ? 
> - Est-ce que "VAE spectral" signifie augmenter les données dans l'espace temporel (signaux), les encoder avec JEPA puis entraîner un MLP ?
> - Et "VAE spectral downstream" signifie-t-il concaténer les embeddings JEPA avec les variables latentes $z$ du VAE pour le classifieur final ? Ou bien "fine-tuner" le générateur pendant la phase downstream ?

## 3. Plan de Vérification
- Vérifier que l'entraînement SSL converge (diminution des termes de variance et covariance, sans effondrement de l'espace latent).
- Exécuter le script de benchmark avec une "seed" fixée pour l'extraction des sous-ensembles (1%, 5%, etc.) afin que chaque stratégie ait exactement les mêmes patients.
- Produire un tableau récapitulatif `results/jepa_ood_benchmark.csv` traçant l'Accuracy et le Recall pour chaque pourcentage et chaque méthode.
