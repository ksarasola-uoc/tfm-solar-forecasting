"""
Módulo de procesamiento de datos para forecasting solar
Autor: [Tu nombre]
Fecha: Enero 2025
Dataset: 6 plantas solares, Sep 2024 – Dic 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from pvlib import solarposition


class SolarDataProcessor:
    """
    Procesador de datos para plantas solares fotovoltaicas.
    
    Funcionalidades:
    - Carga y merge de datos temporales y estáticos
    - Feature engineering (posición solar, variables temporales)
    - Normalización de producción
    - Validación de calidad de datos
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Inicializar procesador.
        
        Args:
            data_dir: Directorio con archivos CSV
        """
        self.data_dir = Path(data_dir)
        self.plantas_info = None
        self.df_completo = None
        
    def load_plantas_info(self, filename: str = 'datos_plantas.csv') -> pd.DataFrame:
        """
        Cargar información estática de plantas.
        
        Returns:
            DataFrame con info de plantas
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"No se encuentra {filepath}")
        
        # Intentar leer con varias codificaciones comunes para evitar
        # UnicodeDecodeError si el archivo no está en UTF-8.
        tried_enc = None
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            tried_enc = 'utf-8'
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding='latin1')
                tried_enc = 'latin1'
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='cp1252')
                    tried_enc = 'cp1252'
                except Exception:
                    # Último recurso: abrir como texto con replacement de caracteres
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        df = pd.read_csv(f)
                    tried_enc = 'utf-8 (errors=replace)'

        if tried_enc:
            print(f"✓ Leído {filepath.name} usando codificación: {tried_enc}")

        # Normalizar columnas numéricas que pueden venir con coma decimal
        if 'instalada_mw' in df.columns:
            # Reemplazar comas por puntos y eliminar caracteres no numéricos
            df['instalada_mw'] = (
                df['instalada_mw']
                .astype(str)
                .str.replace(',', '.', regex=False)
                .str.replace(r'[^0-9.\-+]', '', regex=True)
            )
            df['instalada_mw'] = pd.to_numeric(df['instalada_mw'], errors='coerce')
            if df['instalada_mw'].isnull().any():
                print("⚠️  Aviso: Algunos valores de 'instalada_mw' no pudieron convertirse a numérico (serán NaN)")
        
        # Validar columnas necesarias
        required_cols = ['Nombre', 'Latitud', 'Longitud', 'instalada_mw', 
                        'seguimiento', 'zona_climatica']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en datos_plantas.csv: {missing}")
        
        print(f"✓ Cargada info de {len(df)} plantas")
        print(f"  Distribución por zona climática:")
        print(df['zona_climatica'].value_counts().to_string())
        
        self.plantas_info = df
        return df
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar nombres de columnas: eliminar espacios y unidades entre paréntesis.
        
        Ejemplo: 'temperature_2m (°C)' -> 'temperature_2m'
        
        Args:
            df: DataFrame con columnas a normalizar
            
        Returns:
            DataFrame con columnas normalizadas
        """
        df = df.copy()
        
        # Mapeo de nuevos nombres: eliminar espacios y texto entre paréntesis
        new_cols = {}
        for col in df.columns:
            # Eliminar espacios y todo después del primer '('
            new_col = col.split('(')[0].strip()
            if new_col != col:
                new_cols[col] = new_col
        
        df = df.rename(columns=new_cols)
        return df
    
    def load_temporal_data(self, pattern: str = '*-merged-*.csv') -> pd.DataFrame:
        """
        Cargar datos temporales de todas las plantas.

        Carga todos los archivos que coincidan con el patrón (por defecto ambos
        años: *-merged-2024.csv y *-merged-2025.csv). Para plantas con datos de
        varios años se concatenan y ordenan cronológicamente.

        Args:
            pattern: Patrón de archivos a cargar

        Returns:
            DataFrame concatenado de todas las plantas, ordenado por planta y timestamp
        """
        import re

        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No se encontraron archivos con patrón {pattern}")

        print(f"\nEncontrados {len(files)} archivos de plantas:")

        dfs = []
        for file in sorted(files):
            # Extraer nombre de planta eliminando el sufijo '-merged-YYYY'
            # Ejemplo: "NBALBOA-merged-2024.csv" -> "NBALBOA"
            planta_nombre = re.sub(r'-merged-\d{4}$', '', file.stem)

            df = pd.read_csv(file)
            df['planta_nombre'] = planta_nombre

            # Validar que tenga timestamp
            if 'timestamp_utc' not in df.columns:
                raise ValueError(f"Archivo {file} no tiene columna 'timestamp_utc'")

            # Verificar periodo
            df['timestamp_temp'] = pd.to_datetime(df['timestamp_utc'])
            fecha_min = df['timestamp_temp'].min()
            fecha_max = df['timestamp_temp'].max()

            print(f"  - {planta_nombre} ({file.name}): {len(df):,} registros "
                  f"({fecha_min.date()} a {fecha_max.date()})")

            dfs.append(df)

        df_concat = pd.concat(dfs, ignore_index=True)

        # Ordenar por planta y timestamp, eliminar duplicados por si hubiera solapamiento
        df_concat = df_concat.sort_values(
            ['planta_nombre', 'timestamp_temp']
        ).drop_duplicates(
            subset=['planta_nombre', 'timestamp_utc']
        ).reset_index(drop=True)

        n_plantas = df_concat['planta_nombre'].nunique()
        print(f"\n✓ Total: {len(df_concat):,} registros de {n_plantas} plantas")

        # Normalizar nombres de columnas (elimina unidades entre paréntesis)
        df_concat = self.normalize_column_names(df_concat)

        return df_concat
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validar calidad de datos y reportar estadísticas.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Diccionario con métricas de validación
        """
        print("\n" + "="*60)
        print("VALIDACIÓN DE DATOS")
        print("="*60)
        
        stats = {}
        
        # 1. Valores faltantes
        print("\n1. Valores faltantes por columna:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        has_missing = False
        for col in df.columns:
            if missing[col] > 0:
                print(f"   {col}: {missing[col]:,} ({missing_pct[col]}%)")
                has_missing = True
        
        if not has_missing:
            print("   ✓ No hay valores faltantes")
        
        stats['missing_total'] = missing.sum()
        stats['missing_pct'] = (missing.sum() / (len(df) * len(df.columns)) * 100)
        
        # 2. Valores negativos en producción
        print("\n2. Producción negativa:")
        neg_prod = (df['quantity_sum'] < 0).sum()
        if neg_prod > 0:
            print(f"   ⚠️  {neg_prod:,} registros ({neg_prod/len(df)*100:.2f}%)")
        else:
            print(f"   ✓ No hay producción negativa")
        stats['negative_production'] = neg_prod
        
        # 3. Rangos de variables clave
        print("\n3. Rangos de variables clave:")
        key_vars = ['quantity_sum', 'shortwave_radiation', 
                   'global_tilted_irradiance', 'temperature_2m', 'cloud_cover']
        
        for var in key_vars:
            if var in df.columns:
                print(f"   {var}:")
                print(f"      Min:   {df[var].min():>10.2f}")
                print(f"      Max:   {df[var].max():>10.2f}")
                print(f"      Media: {df[var].mean():>10.2f}")
        
        # 4. Duplicados por timestamp y planta
        print("\n4. Registros duplicados:")
        dups = df.duplicated(subset=['timestamp_utc', 'planta_nombre']).sum()
        if dups > 0:
            print(f"   ⚠️  {dups:,} duplicados")
        else:
            print(f"   ✓ No hay duplicados")
        stats['duplicates'] = dups
        
        # 5. Continuidad temporal
        print("\n5. Continuidad temporal por planta:")
        gap_summary = []
        
        for planta in sorted(df['planta_nombre'].unique()):
            df_p = df[df['planta_nombre'] == planta].copy()
            df_p['timestamp'] = pd.to_datetime(df_p['timestamp_utc'])
            df_p = df_p.sort_values('timestamp')
            
            # Gaps mayores a 1.5 hora
            df_p['diff_hours'] = df_p['timestamp'].diff().dt.total_seconds() / 3600
            gaps = (df_p['diff_hours'] > 1.5).sum()
            
            gap_summary.append({'planta': planta, 'gaps': gaps})
            
            if gaps > 0:
                print(f"   {planta}: {gaps} gaps detectados")
            
            stats[f'gaps_{planta}'] = gaps
        
        total_gaps = sum([x['gaps'] for x in gap_summary])
        if total_gaps == 0:
            print(f"   ✓ Todas las plantas tienen continuidad temporal")
        
        # 6. Cobertura temporal
        print("\n6. Cobertura temporal:")
        df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
        print(f"   Inicio: {df['timestamp'].min()}")
        print(f"   Fin:    {df['timestamp'].max()}")
        print(f"   Días:   {(df['timestamp'].max() - df['timestamp'].min()).days}")
        
        print("\n" + "="*60)
        
        return stats
    
    def merge_data(self, df_temporal: pd.DataFrame, 
                   df_plantas: pd.DataFrame) -> pd.DataFrame:
        """
        Hacer merge de datos temporales con info estática de plantas.
        
        Args:
            df_temporal: Datos temporales
            df_plantas: Info estática de plantas
            
        Returns:
            DataFrame merged
        """
        # Merge por nombre de planta
        df_merged = df_temporal.merge(
            df_plantas,
            left_on='planta_nombre',
            right_on='Nombre',
            how='left'
        )
        
        # Verificar que no haya NaNs tras merge
        plantas_sin_info = df_merged[df_merged['instalada_mw'].isnull()]['planta_nombre'].unique()
        
        if len(plantas_sin_info) > 0:
            print(f"\n⚠️  ADVERTENCIA: Plantas sin info estática:")
            for p in plantas_sin_info:
                print(f"   - {p}")
            raise ValueError(f"Plantas sin match en datos_plantas.csv: {list(plantas_sin_info)}")
        
        print(f"✓ Merge exitoso: {len(df_merged):,} registros")
        print(f"  Plantas integradas: {df_merged['planta_nombre'].nunique()}")
        
        return df_merged
    
    def add_solar_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular posición solar (elevación y azimut).
        
        Args:
            df: DataFrame con timestamp, Latitud, Longitud
            
        Returns:
            DataFrame con elevacion_solar y azimut_solar
        """
        print("\nCalculando posición solar...")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
        
        # Calcular posición para cada planta
        solar_positions = []
        
        for planta in sorted(df['planta_nombre'].unique()):
            mask = df['planta_nombre'] == planta
            df_planta = df[mask]
            
            lat = df_planta['Latitud'].iloc[0]
            lon = df_planta['Longitud'].iloc[0]
            times = df_planta['timestamp']
            
            # Validar coordenadas
            if pd.isna(lat) or pd.isna(lon):
                raise ValueError(f"Planta {planta} no tiene coordenadas válidas")
            
            # Calcular posición solar
            solpos = solarposition.get_solarposition(times, lat, lon)
            
            solar_positions.append(solpos)
        
        # Concatenar todas las posiciones
        all_solpos = pd.concat(solar_positions)
        
        # Añadir al DataFrame original
        df['elevacion_solar'] = all_solpos['elevation'].values
        df['azimut_solar'] = all_solpos['azimuth'].values
        
        # Crear indicador día/noche (elevación > 0 = día)
        df['es_dia'] = (df['elevacion_solar'] > 0).astype(int)
        
        # Estadísticas
        horas_dia = df['es_dia'].sum()
        horas_noche = len(df) - horas_dia
        
        print(f"✓ Posición solar calculada")
        print(f"  Horas diurnas:  {horas_dia:,} ({horas_dia/len(df)*100:.1f}%)")
        print(f"  Horas nocturnas: {horas_noche:,} ({horas_noche/len(df)*100:.1f}%)")
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añadir features temporales derivadas.
        
        Args:
            df: DataFrame con timestamp
            
        Returns:
            DataFrame con features temporales
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
        
        # Features temporales
        df['hora'] = df['timestamp'].dt.hour
        df['dia_semana'] = df['timestamp'].dt.dayofweek
        df['mes'] = df['timestamp'].dt.month
        df['dia_año'] = df['timestamp'].dt.dayofyear
        df['trimestre'] = df['timestamp'].dt.quarter
        df['semana_año'] = df['timestamp'].dt.isocalendar().week
        
        # Indicadores
        df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
        
        # Estación del año (hemisferio norte)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Invierno'
            elif month in [3, 4, 5]:
                return 'Primavera'
            elif month in [6, 7, 8]:
                return 'Verano'
            else:
                return 'Otoño'
        
        df['estacion'] = df['mes'].apply(get_season)
        
        print(f"✓ Features temporales añadidas")
        
        return df
    
    def normalize_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar producción por capacidad instalada.
        
        Args:
            df: DataFrame con quantity_sum e instalada_mw
            
        Returns:
            DataFrame con produccion_norm
        """
        df = df.copy()

        # Asegurar que las columnas numéricas son realmente numéricas
        df['quantity_sum'] = pd.to_numeric(df['quantity_sum'], errors='coerce')
        df['instalada_mw'] = pd.to_numeric(df['instalada_mw'], errors='coerce')

        # Evitar división por cero o por NaN: reemplazar 0/NaN en instalada_mw por np.nan
        df.loc[df['instalada_mw'] == 0, 'instalada_mw'] = np.nan

        # quantity_sum está en MWh (producción horaria)
        # instalada_mw es la capacidad en MW
        # produccion_norm = MWh / MW = horas equivalentes a plena carga
        df['produccion_norm'] = df['quantity_sum'] / df['instalada_mw']
        
        # Validar rangos razonables
        # Una planta solar en hora pico puede producir ~0.8-1.0 h equivalentes
        # (factor de capacidad instantáneo cercano a 100%)
        max_prod_norm = df['produccion_norm'].max()
        
        if max_prod_norm > 1.5:
            print(f"⚠️  ADVERTENCIA: Producción normalizada máxima = {max_prod_norm:.2f}")
            print(f"   (esperado: <1.0 para solar)")
        
        print(f"✓ Producción")
        print(f"   Min:   {df['produccion'].min():.4f} h equiv.")
        print(f"   Max:   {df['produccion'].max():.4f} h equiv.")
        print(f"   Media: {df['produccion'].mean():.4f} h equiv.")
        
        return df
    
    def clean_outliers(self, df: pd.DataFrame, 
                       threshold_std: float = 4.0) -> pd.DataFrame:
        """
        Limpiar outliers extremos en producción.
        
        Args:
            df: DataFrame
            threshold_std: Umbral en desviaciones estándar
            
        Returns:
            DataFrame limpio
        """
        df = df.copy()
        
        print(f"\nLimpieza de outliers (>{threshold_std} std)...")
        
        outliers_total = 0
        
        for planta in sorted(df['planta_nombre'].unique()):
            mask_planta = df['planta_nombre'] == planta
            
            # Solo para horas diurnas (donde hay producción esperada)
            mask_dia = df['es_dia'] == 1
            mask = mask_planta & mask_dia
            
            if mask.sum() == 0:
                continue
            
            mean = df.loc[mask, 'produccion'].mean()
            std = df.loc[mask, 'produccion'].std()
            
            # Identificar outliers
            mask_outlier = (
                mask & 
                ((df['produccion'] > mean + threshold_std * std) |
                 (df['produccion'] < mean - threshold_std * std))
            )
            
            n_outliers = mask_outlier.sum()
            outliers_total += n_outliers
            
            if n_outliers > 0:
                print(f"   {planta}: {n_outliers} outliers")
                
                # Imputar con interpolación
                df.loc[mask_outlier, 'produccion'] = np.nan
        
        if outliers_total == 0:
            print(f"   ✓ No se detectaron outliers")
        else:
            # Interpolar valores faltantes
            df['produccion'] = df.groupby('planta_nombre')['produccion'].transform(
                lambda x: x.interpolate(method='linear', limit=3)
            )
            
            print(f"✓ {outliers_total} outliers corregidos mediante interpolación")
        
        return df
    
    def process_pipeline(self) -> pd.DataFrame:
        """
        Pipeline completo de procesamiento.
        
        Returns:
            DataFrame procesado y listo para modelado
        """
        print("\n" + "="*80)
        print("PIPELINE DE PROCESAMIENTO DE DATOS")
        print("Dataset: 6 plantas solares, Sep 2024 – Dic 2025")
        print("="*80)
        
        # 1. Cargar datos
        print("\n[1/7] Cargando datos...")
        df_plantas = self.load_plantas_info()
        df_temporal = self.load_temporal_data()
        
        # 2. Validar datos crudos
        print("\n[2/7] Validando datos crudos...")
        stats = self.validate_data(df_temporal)
        
        # 3. Merge
        print("\n[3/7] Haciendo merge de datos...")
        df = self.merge_data(df_temporal, df_plantas)

        # 4. Renombrar columna quantity_sum a produccion
        df = df.rename(columns={'quantity_sum': 'produccion'})
        print("\n✓ Columna 'quantity_sum' renombrada a 'produccion'")
        
        
        # 5. Calcular posición solar
        print("\n[4/7] Calculando posición solar...")
        df = self.add_solar_position(df)
        
        # 6. Features temporales
        print("\n[5/7] Añadiendo features temporales...")
        df = self.add_temporal_features(df)
        
        # 6. Normalizar producción
        #print("\n[6/7] Normalizando producción...")
        #df = self.normalize_production(df)
        
        # 7. Limpiar outliers
        print("\n[7/7] Limpiando outliers...")
        df = self.clean_outliers(df)
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETADO")
        print("="*80)
        print(f"Dataset final:")
        print(f"  Registros: {len(df):,}")
        print(f"  Columnas:  {len(df.columns)}")
        print(f"  Plantas:   {df['planta_nombre'].nunique()}")
        print(f"  Periodo:   {df['timestamp'].min().date()} a {df['timestamp'].max().date()}")
        # Asegurarse de convertir a str y eliminar NaNs antes de hacer join
        if 'zona_climatica' in df.columns:
            zonas = df['zona_climatica'].dropna().astype(str).unique()
            zonas_str = ', '.join(zonas) if len(zonas) > 0 else 'N/A'
        else:
            zonas_str = 'N/A'
        print(f"  Zonas:     {zonas_str}")
        
        
        self.df_completo = df
        
        return df
    
    def split_train_val_test(self, df: pd.DataFrame,
                             train_end: str = '2025-09-12',
                             val_end: str = '2025-10-31') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Dividir dataset en train/val/test temporalmente.
        
        División temporal fija (independiente del volumen total de datos):
        - Train: Sep 2024 – 12 Sep 2025  (todo lo anterior al corte)
        - Val:   13 Sep – 31 Oct 2025
        - Test:  1 Nov – 31 Dic 2025     (periodo de evaluación fijo)
        
        Args:
            df: DataFrame procesado
            train_end: Fecha fin de entrenamiento (YYYY-MM-DD)
            val_end: Fecha fin de validación (YYYY-MM-DD)
            
        Returns:
            Tupla (df_train, df_val, df_test)
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp_utc'])
        
        # Convertir fechas a datetime
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)
        
        # Si los timestamps del dataframe son tz-aware, hacer que las fechas de
        # comparación también lo sean (tz-aware) para evitar TypeError
        if df['timestamp'].dtype.tz is not None:
            train_end_dt = train_end_dt.tz_localize('UTC')
            val_end_dt = val_end_dt.tz_localize('UTC')
        
        # Crear máscaras
        mask_train = df['timestamp'] <= train_end_dt
        mask_val = (df['timestamp'] > train_end_dt) & (df['timestamp'] <= val_end_dt)
        mask_test = df['timestamp'] > val_end_dt
        
        df_train = df[mask_train].copy()
        df_val = df[mask_val].copy()
        df_test = df[mask_test].copy()
        
        # Calcular porcentajes
        pct_train = len(df_train) / len(df) * 100
        pct_val = len(df_val) / len(df) * 100
        pct_test = len(df_test) / len(df) * 100
        
        print("\n" + "="*80)
        print("DIVISIÓN TEMPORAL DEL DATASET (2024-2025)")
        print("="*80)
        print(f"\nTrain: {len(df_train):,} registros ({pct_train:.1f}%)")
        print(f"  Periodo: {df_train['timestamp'].min().date()} a {df_train['timestamp'].max().date()}")
        print(f"  Meses:   {(df_train['timestamp'].max() - df_train['timestamp'].min()).days / 30.4:.1f}")
        
        print(f"\nVal:   {len(df_val):,} registros ({pct_val:.1f}%)")
        print(f"  Periodo: {df_val['timestamp'].min().date()} a {df_val['timestamp'].max().date()}")
        print(f"  Meses:   {(df_val['timestamp'].max() - df_val['timestamp'].min()).days / 30.4:.1f}")
        
        print(f"\nTest:  {len(df_test):,} registros ({pct_test:.1f}%)")
        print(f"  Periodo: {df_test['timestamp'].min().date()} a {df_test['timestamp'].max().date()}")
        print(f"  Meses:   {(df_test['timestamp'].max() - df_test['timestamp'].min()).days / 30.4:.1f}")
        
        print("\n" + "-"*80)
        print("NOTA METODOLÓGICA:")
        print("División temporal fija: val=Sep-Oct 2025, test=Nov-Dic 2025.")
        print("El train incluye datos desde Sep 2024 para plantas con datos de 2024.")
        print("UPGARNACHA solo dispone de datos 2025 (limitación documentada).")
        print("-"*80)
        
        # Validar que no haya solapamiento
        assert len(set(df_train.index) & set(df_val.index)) == 0, "Solapamiento train-val"
        assert len(set(df_train.index) & set(df_test.index)) == 0, "Solapamiento train-test"
        assert len(set(df_val.index) & set(df_test.index)) == 0, "Solapamiento val-test"
        
        return df_train, df_val, df_test