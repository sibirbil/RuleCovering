# # -*- coding: utf-8 -*-

import pandas as pd

def banknote(wd): # Two classes
    """
    Attribute Information:
    1. variance of Wavelet Transformed image (continuous)
    2. skewness of Wavelet Transformed image (continuous)
    3. curtosis of Wavelet Transformed image (continuous)
    4. entropy of image (continuous)
    5. class (integer)
    """
    df = pd.read_csv(wd+'data_banknote_authentication.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df
    
def ILPD(wd): # Two classes
    """
    Attribute Information:
    1. Age Age of the patient
    2. Gender Gender of the patient
    3. TB Total Bilirubin
    4. DB Direct Bilirubin
    5. Alkphos Alkaline Phosphotase
    6. Sgpt Alamine Aminotransferase
    7. Sgot Aspartate Aminotransferase
    8. TP Total Protiens
    9. ALB Albumin
    10. A/G Ratio Albumin and Globulin Ratio
    11. Selector field used to split the data into 
        two sets (labeled by the experts)
    """
    df = pd.read_csv(wd+'ILPD.csv',header = None)
    df.iloc[:,1] = (df.iloc[:,1] == 'Female')*1
    df.iloc[:,-1] = (df.iloc[:,-1] == 2) * 1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df.dropna(inplace=True)
    return df

def ionosphere(wd): # Two classes
    """
    Attribute Information:

    -- All 34 are continuous
    -- The 35th attribute is either "good" or "bad" 
       according to the definition summarized above. 
       This is a binary classification task.
    """
    df = pd.read_csv(wd+'ionosphere.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 'g')*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def transfusion(wd): # Two classes
    """
    Attribute Information:
    Given is the variable name, variable type, the measurement unit and a 
    brief description. The "Blood Transfusion Service Center" is a 
    classification problem. The order of this listing corresponds to the 
    order of numerals along the rows of the database.

    R (Recency - months since last donation),
    F (Frequency - total number of donation),
    M (Monetary - total blood donated in c.c.),
    T (Time - months since first donation), and
    a binary variable representing whether he/she donated blood in March 2007 
    (1 stand for donating blood; 0 stands for not donating blood).
    """
    df = pd.read_csv(wd+'transfusion.csv', header = 0)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def liver(wd): # Two classes
    """
    Attribute information:
    1. mcv	mean corpuscular volume
    2. alkphos	alkaline phosphotase
    3. sgpt	alamine aminotransferase
    4. sgot 	aspartate aminotransferase
    5. gammagt	gamma-glutamyl transpeptidase
    6. drinks	number of half-pint equivalents of 
       alcoholic beverages drunk per day
    7. selector  field used to split data into two set
   """
    df = pd.read_csv(wd+'bupa.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 2)*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def tictactoe(wd): # Two classes
    """
    Attribute Information:
    1. top-left-square: {x,o,b}
    2. top-middle-square: {x,o,b}
    3. top-right-square: {x,o,b}
    4. middle-left-square: {x,o,b}
    5. middle-middle-square: {x,o,b}
    6. middle-right-square: {x,o,b}
    7. bottom-left-square: {x,o,b}
    8. bottom-middle-square: {x,o,b}
    9. bottom-right-square: {x,o,b}
    10. Class: {positive,negative}
    """
    df = pd.read_csv(wd+'tictactoe.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df1 = pd.get_dummies(df.iloc[:,:-1], drop_first=True)
    df1['y'] = (df['y'] == 'positive') *1
    return df1

def wdbc(wd): # Two classes
    """
    1) ID number
    2) Diagnosis (M = malignant, B = benign)
    3-32)
    Ten real-valued features are computed for each cell nucleus:
        a) radius (mean of distances from center to points on the perimeter)
        b) texture (standard deviation of gray-scale values)
        c) perimeter
        d) area
        e) smoothness (local variation in radius lengths)
        f) compactness (perimeter^2 / area - 1.0)
        g) concavity (severity of concave portions of the contour)
        h) concave points (number of concave portions of the contour)
        i) symmetry 
        j) fractal dimension ("coastline approximation" - 1)
    """
    df = pd.read_csv(wd+'wdbc.csv', header = None, index_col = 0)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = (df['y'] == 'M')*1
    df.drop('y', axis=1, inplace = True)
    df['y'] = y
    return df

def mammography(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    7. Class (-1 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'mammography.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def diabetes(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'diabetes.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def oilspill(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    x. Class (0 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'oilspill.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df = df.drop(df.columns[[0]], axis = 1)
    return df

def phoneme(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    Five different attributes were chosen to
    characterize each vowel: they are the amplitudes of the five first
    harmonics AHi, normalised by the total energy Ene (integrated on all the
    frequencies): AHi/Ene. Each harmonic is signed: positive when it
    corresponds to a local maximum of the spectrum and negative otherwise.
    6. Class (0 and 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'phoneme.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def seeds(wd): # Three classes
    """
    Attribute Information:

    To construct the data, seven geometric parameters of wheat kernels were measured:
    1. area A,
    2. perimeter P,
    3. compactness C = 4*pi*A/P^2,
    4. length of kernel,
    5. width of kernel,
    6. asymmetry coefficient
    7. length of kernel groove.
    All of these parameters were real-valued continuous.
    """
    df = pd.read_csv(wd+'seeds.csv', header = None, sep = '\t', engine = 'python')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def wine(wd): # Three classes
    """
    The attributes are donated by Riccardo Leardi (riclea@anchem.unige.it)
 	 1) Alcohol
 	 2) Malic acid
 	 3) Ash
	 4) Alcalinity of ash  
 	 5) Magnesium
	 6) Total phenols
 	 7) Flavanoids
 	 8) Nonflavanoid phenols
 	 9) Proanthocyanins
	10) Color intensity
 	11) Hue
 	12) OD280/OD315 of diluted wines
 	13) Proline            
    Number of Instances
    class 1 59
	class 2 71
	class 3 48
    """
    df = pd.read_csv(wd+'wine.csv', header = None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = df['y']
    df.drop('y', axis = 1, inplace = True)
    df['y'] = y
    return df

def glass(wd): # Six classes - Imbalanced
    """
    Attribute Information:
    RI: refractive index
    Na: Sodium
    Mg: Magnesium
    Al: Aluminum
    Si: Silicon
    K: Potassium
    Ca: Calcium
    Ba: Barium
    Fe: Iron

    Class 1: building windows (float processed)
    Class 2: building windows (non-float processed)
    Class 3: vehicle windows (float processed)
    Class 4: containers
    Class 5: tableware
    Class 6: headlamps

    """
    df = pd.read_csv(wd+'glass.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    df['y'] -= 1
    return df

def ecoli(wd): # Eight classes - Imbalanced
    """
    Attribute Information:
    0: McGeoch’s method for signal sequence recognition
    1: von Heijne’s method for signal sequence recognition
    2: von Heijne’s Signal Peptidase II consensus sequence score
    3: Presence of charge on N-terminus of predicted lipoproteins
    4: Score of discriminant analysis of the amino acid content 
       of outer membrane and periplasmic proteins.
    5: score of the ALOM membrane-spanning region prediction program
    6: score of ALOM program after excluding putative cleavable 
       signal regions from the sequence.
    
    Eight Classes:
    0: cytoplasm
    1: inner membrane without signal sequence
    2: inner membrane lipoprotein
    3: inner membrane, cleavable signal sequence
    4: inner membrane, non cleavable signal sequence
    5: outer membrane
    6: outer membrane lipoprotein
    7: periplasm
    """
    df = pd.read_csv(wd+'ecoli.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df