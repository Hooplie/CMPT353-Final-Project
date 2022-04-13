import pandas as pd
import csv
from itertools import islice
import matplotlib.pyplot as plt
import math

RAW_DATA_FIlE = "./Data/data.txt"
RAW_FORMAT_FILE = "./Layouts/SAS/CIS2018_PUMF_frq.SAS"
FORMAT_FILE = "formatted.csv"
DATA_FILE = "data.csv"
NULL_VALS_FILE = "./Layouts/SPSS/CIS2018_PUMF_miss.sps"
MISSING_VAL = 999999
MAX_MISS_VAL = 17188

M1 = ('PROV', 'USZGAP', 'MBMREGP', 'AGEGP', 'MARSTP', 'SCSUM', 'ALFST', 'WKSEM', 'WKSUEM', 'WKSNLF', 'MAJRI', 'EFSIZE', 
'EFTYP', 'EFAGOFMP', 'EFAGYFMP', 'EFMJSI', 'CFSIZE', 'CFCOMP', 'CFAGOFMP', 'CFAGYFMP', 'CFMJSI', 'HHSIZE')

M2 = ('SEX', 'CMPHI', 'HLEV2G', 'STUDTFP', 'FLLPRTP', 'FWORKED', 'FPDWK', 'FSEMP', 'FUNFW', 'IMMST', 'YRIMMG', 'EFMJIE', 
'EFMJIEH', 'EFRMJIG', 'CFMJIE', 'CFMJIEH', 'CFRMJIG', 'HHCOMP', 'HHMJIE', 'LICOFA', 'LICOFB', 'LIMSFA', 'MBSCF18', 
'DWLTYP', 'DWTENR', 'REPA', 'SUIT', 'MORTG', 'RNSUB', 'CHNEED', 'FSCADLTM', 'FSCCHLDM', 'FSCHHLDM')

M3 = ('USHRWK', 'LICODA', 'LICODB', 'LIMSDA', 'MBSCD18')

M4 = ('ALHRWK', 'ALHRWK')

M5 = ('ALIMO', 'ALIP', 'ATINC', 'CAPGN', 'CCAR', 'CHFED', 'CHPRV', 'CHTXB', 'CPQPP', 'CQPC', 'EARNG', 'EIPR', 'FDITX', 'GI', 
'GSTXC', 'GTR', 'INCTX', 'INVA', 'MTINC', 'OAS', 'OASGI', 'OGOVTR', 'OTHINC', 'OTTXM', 'PEN', 'PENGIV', 'PENREC', 'PHPR', 
'PRPEN', 'PVITX', 'PVTXC', 'RPPC', 'RSPWI', 'SAPIS', 'SEMP', 'TTINC', 'UDPD', 'UIBEN', 'WGSAL', 'WKRCP', 'EFALIMO', 
'EFALIP', 'EFATINC', 'EFCAPGN', 'EFCCAR', 'EFCHFED', 'EFCHPRV', 'EFCHTXB', 'EFCPQPP', 'EFCQPC', 'EFEARNG', 'EFEIPR', 
'EFFDITX', 'EFGI', 'EFGSTXC', 'EFGTR', 'EFINCTX', 'EFINVA', 'EFMBIN18', 'EFMTINC', 'EFOAS', 'EFOASGI', 'EFOGOVTR', 
'EFOTHINC', 'EFOTTXM', 'EFPEN', 'EFPENGIV', 'EFPENREC', 'EFPHPR', 'EFPRPEN', 'EFPVITX', 'EFPVTXC', 'EFRPPC', 'EFRSPWI', 
'EFSAPIS', 'EFSEMP', 'EFTTINC', 'EFUDPD', 'EFUIBEN', 'EFWGSAL', 'EFWKRCP', 'CFALIMO', 'CFALIP', 'CFATINC', 'CFCAPGN', 
'CFCCAR', 'CFCHFED', 'CFCHPRV', 'CFCHTXB', 'CFCPQPP', 'CFCQPC', 'CFEARNG', 'CFEIPR', 'CFFDITX', 'CFGI', 'CFGSTXC', 'CFGTR', 
'CFINCTX', 'CFINVA', 'CFMTINC', 'CFOAS', 'CFOASGI', 'CFOGOVTR', 'CFOTHINC', 'CFOTTXM', 'CFPEN', 'CFPENGIV', 'CFPENREC', 
'CFPHPR', 'CFPRPEN', 'CFPVITX', 'CFPVTXC', 'CFRPPC', 'CFRSPWI', 'CFSAPIS', 'CFSEMP', 'CFTTINC', 'CFUDPD', 'CFUIBEN', 
'CFWGSAL', 'CFWKRCP', 'MORTGM', 'CONDMP', 'RENTM')

M_DICT = {M1:[96, 99], M2:[6, 9], M3:[999.6, 999.9], M4:[9996, 9999], M5:[99999996, 99999999]}

def write_file(rows, line_vector, output_file):
    f = open(output_file, 'w')
    writer = csv.writer(f)
    for i in range(len(line_vector)-1):
        row = []
        temp = line_vector[i].split()
        for j in rows:
            row.append(temp[j])
        writer.writerow(row)
    f.close()

def extract_format(input_file, output_file):
    # determine the offsets of the file to pass to read_fwf()
    filename = input_file
    rows = [2,6,8]
    f = open(filename, mode='r', encoding='utf8', newline='\r\n')
    line_vector = []
    for line in islice(f, 2, 195):
        line_vector.append(line)
    f.close()
    write_file(rows, line_vector, output_file)

def read_data(format_file, data_file, output_file):
    # read the data with the field widths in offsets
    offsets = pd.read_csv(format_file, header=None)
    colspecs = []
    for index, row in offsets.iterrows():
        start = row[1]-1
        end = row[2]
        colspecs.append((start,end))
    df = pd.read_fwf(data_file, colspecs=colspecs, header=None, index_col=0)
    # format the data
    df = df.reset_index()
    df.columns = offsets.loc[:, 0]
    df.to_csv(output_file, index=False)

def process_null_vals(null_vals_file):
    null_vals = []
    # tokenize the file containing the null values
    with open(null_vals_file) as file:
        for line in islice(file, 2, 141):
            null_vals.append(line.split())
    # format the dataframe to useable values
    null_df = pd.DataFrame(null_vals, columns=['label', 'start', 'thru', 'end'])
    null_df = null_df.drop('thru', axis=1)
    null_df['start'] = null_df['start'].str[1:].astype(float)
    null_df['end'] = null_df['end'].str[:-1].astype(float)
    return null_df

def remove_missing_vals(data_file, null_vals_file, output_file):
    # use the dataframe containing missing value codes on our data
    null_vals = process_null_vals(null_vals_file)
    data = pd.read_csv(data_file)
    missing = []
    # create the m# lists consisting of missing values not in the null_vals_file
    # for label in data.columns: 
    #     if label not in null_vals['label']:
    #         missing.append(label)
    # print(missing)
    for index, row in null_vals.iterrows():
        # change missing values to a number that is not found in the data
        data.loc[(data[row['label']] >= row['start']) & (data[row['label']] <= row['end']), row['label']] = math.nan
        # drop columns that have too many missing values
        # sizes = data.groupby(row['label']).size().to_frame()
        # if MISSING_VAL in sizes.index:
        #     if sizes.loc[MISSING_VAL][0] >= MAX_MISS_VAL:
        #         data = data.drop(columns=row['label'])
    for key, value in M_DICT.items():
        for k in key:
            data.loc[(data[k] >= value[0]) & (data[k] <= value[1]), k] = math.nan
    data.to_csv(output_file, index=False)

def plot_boxplots(data_file):
    data = pd.read_csv(data_file)
    data = data.astype(float)
    dim = data.shape[1]
    dim = math.ceil(math.sqrt(dim)/4)
    figures = []
    axes = []
    labels = data.columns
    i = 0
    for j in range(16):
        figures.append(f'f{j}')
        axes.append(f'a{j}')
        figures[j], axes[j] = plt.subplots(dim, dim)
        for axes[j] in axes[j].reshape(-1): 
            if i < data.shape[1]:
                axes[j].boxplot(data[labels[i]])
                axes[j].set_title(labels[i])
            i += 1
    plt.show()

def preproccess():
    extract_format(RAW_FORMAT_FILE, FORMAT_FILE)
    read_data(FORMAT_FILE, RAW_DATA_FIlE, DATA_FILE)
    # every entry had at least 1 missing value... removing missing values results in no data points.
    remove_missing_vals(DATA_FILE, NULL_VALS_FILE, DATA_FILE)
    # plot_boxplots(DATA_FILE)

