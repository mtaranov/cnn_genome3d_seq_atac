import numpy as np
import pybedtools
from pybedtools import BedTool
import csv
import copy
from sklearn.model_selection import train_test_split

def bed_intersection(bed1_in, bed2_in, bed_out=False, bed_out_name='',  f=0, F=0, e=True):
    """
    intersects bed1 with bed2 
    """
  
    bed1 = BedTool(bed1_in)
    bed2 = BedTool(bed2_in)
    
    if bed_out == True:
        bed1.intersect(bed2_in, c=True, f=f, F=F, e=e).saveas(bed_out_name)
    return bed1.intersect(bed2_in, c=True, f=f, F=F, e=e)  

def bed(bed1_in, bed2_in, bed_out=False, bed_out_name='',  f=0, F=0, e=True, wo=True):
    """
    intersects bed1 with bed2 
    """
  
    bed1 = BedTool(bed1_in)
    bed2 = BedTool(bed2_in)
    
    if bed_out == True:
        bed1.intersect(bed2_in, f=f, F=F, e=e, wo=wo).saveas(bed_out_name)
    return bed1.intersect(bed2_in, f=f, F=F, e=e, wo=wo)   

def featuretype_filter(feature, featuretype):
    """
    returns lines which match featuretype 
    """
    if int(feature[5]) == featuretype:
        return True
    return False

def bed_w_NoIntersection(bed1_in, bed2_in,  bed_out_NoInters, bed_out=False, bed_out_name='',featuretype=0, f=0, F=0, e=True):
    """
    filters based on the featuretype_filter
    """
    a=bed_intersection(bed1_in, bed2_in, bed_out=bed_out, f=f, F=F, e=e)
    result=a.filter(featuretype_filter, featuretype).saveas(bed_out_NoInters)
    return pybedtools.BedTool(result.fn)


def bed_closest(bed1_in, bed2_in, bed_out=False, bed_out_name='', io=True):
    """
    finds closest distance
    """

    bed1 = BedTool(bed1_in)
    bed2 = BedTool(bed2_in)

    if bed_out == True:
        bed1.closest(bed2, io=io).saveas(bed_out_name)
    return bed1.closest(bed2, io=io)

# pulls bait-bait contacts from all-all capture  contacts
def get_bait_bait_contacts(InteractionsFile_D0, PromoterCaptureFile, outFile, site1, site2):
    data = BedTool(InteractionsFile_D0)
    BedTool([i[0], i[1], i[2], i[6]] for i in data).saveas(site1)
    BedTool([i[3], i[4], i[5], i[6]] for i in data).saveas(site2)
    bed1_site1=BedTool(site1)
    bed1_site2=BedTool(site2)
    bed2 = BedTool(PromoterCaptureFile)

    with open(outFile, 'w') as outcsv:
        #configure writer to write standart csv file
            writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for interval in zip(bed1_site1.intersect(bed2, wao=True), bed1_site2.intersect(bed2, wao=True)):
                if interval[0][0] == interval[1][0]: # intra-chr contacts only 
                    if (int(interval[0][9]) != 0 and int(interval[1][9]) != 0):
                        writer.writerow([interval[0][0], interval[0][1], interval[0][2], interval[1][0], interval[1][1], interval[1][2], interval[1][3], interval[0][7], interval[1][7]])

# intersects atac-peak file with promoter-capture file
# assigns each peak summit to a HindIII
# outputs 1kb regions aroud peak summit and corresponding HindIII
def get_atac_bins_at_1kb(bed1_in, bed2_in, outFile, bed_out=False, bed_out_name='',  f=0.0, F=0.0, e=True, wo=True):

    bed1 = BedTool(bed1_in)
    bed2 = BedTool(bed2_in)

    if bed_out == True:
        bed1.intersect(bed2, f=f, F=F, e=e, wo=wo).saveas(bed_out_name)
    
    atac_HindIII={}
    # if one ATAC-fragment/peak overlaps multiple HindIII fragments, keep ATAC-fragment w the largest overlap
    for interval in bed1.intersect(bed2, f=f, F=F, e=e, wo=wo):
        if interval[0] not in atac_HindIII:
            atac_HindIII[interval[0]]={interval[1] : {interval[9] : [interval[13], interval[15]]}}
        else:
            if interval[1] not in atac_HindIII[interval[0]]:
                atac_HindIII[interval[0]][interval[1]]={interval[9] : [interval[13], interval[15]]}
            else:
                if interval[9] not in atac_HindIII[interval[0]][interval[1]]:
                     atac_HindIII[interval[0]][interval[1]][interval[9]]=[interval[13], interval[15]]
                else:
                     if interval[15] > atac_HindIII[interval[0]][interval[1]][interval[9]][1]:
                        atac_HindIII[interval[0]][interval[1]].update({interval[9] : [interval[13], interval[15]]})          
                    
    with open(outFile, 'w') as outcsv:
    #configure writer to write standart csv file
        writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for chr in atac_HindIII.keys():
            for start_pos in atac_HindIII[chr].keys():
                for peaksummit in atac_HindIII[chr][start_pos].keys():
                    
                    writer.writerow([str(chr), str(int(start_pos) + int(peaksummit) - 500),  str(int(start_pos) + int(peaksummit) + 500),  str(atac_HindIII[chr][start_pos][peaksummit][0])])
                
# get the list of HindIII contact pairs for each chr
def get_HindIII_ContactPairs(bait_bait_file):
    contact_dict={}
    for line in open(bait_bait_file,'r'):
        words=line.rstrip().split()
        chr1=words[0]
        chr2=words[3]
        q_value=words[6]
        HindIII1=words[7]
        HindIII2=words[8]
        if chr1 != chr2:
            raise ValueError("Inter-chr contact!")
            exit
        else:
            if float(q_value) > 0: # only confident contacts
                if chr1 not in contact_dict:
                    if int(HindIII1) < int(HindIII2):
                        contact_dict[chr1]=[(HindIII1, HindIII2)]
                    else:
                        contact_dict[chr1]=[(HindIII2, HindIII1)]
                else:
                    if int(HindIII1) < int(HindIII2):
                        contact_dict[chr1].append((HindIII1, HindIII2))
                    else:
                        contact_dict[chr1].append((HindIII2, HindIII1))
    return contact_dict

# assigns 1/0 labels to atac_bin interactions
def get_2D_atac_labels(atac_bins_file, pos_HindIII_pairs, outFile):
    atac_dict={}
    label_list=[]
    
    for line in open(atac_bins_file,'r'):
        words=line.rstrip().split()
        chr=words[0]
        start=words[1]
        end=words[2]
        HindIII=words[3]
        if chr not in atac_dict:
            atac_dict[chr]=[(start, end, HindIII)]
        else:
            atac_dict[chr].append((start, end, HindIII))

    with open(outFile, 'w') as outcsv:
    #configure writer to write standart csv file
        writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        label_list=[]
        for chr in atac_dict:
            #if chr=='chrX':
            for item1 in atac_dict[chr]:
                for item2 in atac_dict[chr]:   
                    if int(item1[0]) < int(item2[0]):
                        if ((item1[2],item2[2]) in pos_HindIII_pairs[chr]):
                            label=1
                        else:
                            label=0
                        label_list.append(label)
                        writer.writerow([chr, item1[0],item1[1], chr, item2[0],item2[1], label, item1[2],item2[2]])
    return label_list

def get_contacts_at_distance(atac_w_labels_file, thres_min, thres_max, outFile):
    atac_dict={}
    for line in open(atac_w_labels_file,'r'):
        words=line.rstrip().split()
        chr=words[0]
        start1=words[1]
        end1=words[2]
        start2=words[4]
        end2=words[5]
        label=words[6]
        HindIII1=words[7]
        HindIII2=words[8]
        if (abs(int(start2)-int(start1)) > thres_min and abs(int(start2)-int(start1)) < thres_max):
            if chr not in atac_dict:
                atac_dict[chr]=[(start1, end1, start2, end2, label, HindIII1, HindIII2)]
            else:
                atac_dict[chr].append((start1, end1, start2, end2, label, HindIII1, HindIII2))

    with open(outFile, 'w') as outcsv:
    #configure writer to write standart csv file
        writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for chr in atac_dict:
                for item in atac_dict[chr]:
                    writer.writerow([chr, item[0], item[1], chr, item[2], item[3], item[4], item[5], item[6]])

def parse_interval_pairs_and_labels(interval_pairs_and_labels_bed, save_to_prefix):
    """
    Takes bed file with interval pairs, returns two separate BedTools
    """
    interval_pairs_df = BedTool(interval_pairs_and_labels_bed).to_dataframe()
    intervals_side1 = BedTool.from_dataframe(interval_pairs_df.iloc[:,[0,1,2]])
    intervals_side2 = BedTool.from_dataframe(interval_pairs_df.iloc[:,[3,4,5]])
    hindIII_frag_ids = BedTool.from_dataframe(interval_pairs_df.iloc[:,[0,7,8]])
    labels = np.asarray(interval_pairs_df.iloc[:,[6]], dtype=int)
    # save to files
    intervals_side1.saveas("{}.intervals_side1.bed".format(save_to_prefix))
    intervals_side2.saveas("{}.intervals_side2.bed".format(save_to_prefix))
    np.save("{}.labels.npy".format(save_to_prefix), labels)
    hindIII_frag_ids.saveas("{}.fragment_ids.bed".format(save_to_prefix))

    return (intervals_side1, intervals_side2, labels)

def get_pairs_distance_matched(X, y, indx, min_dist, max_dist, dist_step, imbalance_ratio):

    def subsample_indx(indecies, size, imbalance_ratio):
        indecies_shuffled=copy.copy(indecies)
        np.random.shuffle(indecies_shuffled)
        num_subsampled = size*imbalance_ratio
        if num_subsampled > len(indecies[0]):
            print '    Error: Not enough to subsample'
            exit
        #print indecies_shuffled[0].shape
        #print indecies_shuffled[0][:num_subsampled].shape
        else:
            return indecies_shuffled[0][:num_subsampled]

    neg_indxs = np.where(y.astype(float)==0)[0]
    pos_indxs = np.where(y.astype(float)>=10)[0]
    X_pos=X[pos_indxs]
    X_neg=X[neg_indxs]
    y_pos=y[pos_indxs]
    y_neg=y[neg_indxs]
    indx_pos=indx[pos_indxs]
    indx_neg=indx[neg_indxs]

    thres1=min_dist+dist_step
    thres2=min_dist

    X_new=np.empty(([0,X.shape[1]]))
    y_new=np.empty(([0,y.shape[1]]))
    indx_new=np.empty(([0,indx.shape[1]]))

    while thres1 <= max_dist:
        print 'distance window: ', '[', thres2, ',', thres1, ']'
        neg_indx_at_dist=np.where((abs(X_neg[:,1].astype(int) - X_neg[:,4].astype(int)) <= thres1) & (abs(X_neg[:,1].astype(int) - X_neg[:,4].astype(int)) >= thres2))
        pos_indx_at_dist=np.where((abs(X_pos[:,1].astype(int) - X_pos[:,4].astype(int)) <= thres1) & (abs(X_pos[:,1].astype(int) - X_pos[:,4].astype(int)) >= thres2))
        if len(pos_indx_at_dist[0])> len(neg_indx_at_dist[0]):
            #print 'more pos than neg'
            print 'number of pos at distance=:',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=:',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(pos_indx_at_dist, len(neg_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=indx_subsampled
            new_neg_indx_at_dist=neg_indx_at_dist[0]

        else:
            #print 'more neg than pos'
            print 'number of pos at distance=',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(neg_indx_at_dist, len(pos_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=pos_indx_at_dist[0]
            new_neg_indx_at_dist=indx_subsampled

        y_pos_at_dist=y_pos[new_pos_indx_at_dist]
        y_neg_at_dist=y_neg[new_neg_indx_at_dist]
        X_pos_at_dist=X_pos[new_pos_indx_at_dist]
        X_neg_at_dist=X_neg[new_neg_indx_at_dist]
        indx_pos_at_dist=indx_pos[new_pos_indx_at_dist]
        indx_neg_at_dist=indx_neg[new_neg_indx_at_dist]

        y_at_dist=np.concatenate((y_pos_at_dist, y_neg_at_dist))
        X_at_dist=np.concatenate((X_pos_at_dist, X_neg_at_dist))
        indx_at_dist=np.concatenate((indx_pos_at_dist, indx_neg_at_dist))

        print 'labels at dist: ', y_at_dist.shape
        print 'data at dist: ', X_at_dist.shape
        print 'indx at dist: ', indx_at_dist.shape

        indx_new=np.concatenate((indx_new, indx_at_dist))
        X_new=np.concatenate((X_new, X_at_dist))
        y_new=np.concatenate((y_new, y_at_dist))


        #print X_new.shape, X_at_dist.shape
        #print y_new.shape, y_at_dist.shape
        #print indx_new.shape, indx_at_dist.shape

        #print "# of neg:", np.where(y_at_dist==0)[0].shape
        #print "# of pos:", np.where(y_at_dist==1)[0].shape
        #thres2=thres1+min_dist
        thres2=thres1
        thres1=thres1+dist_step

    return X_new, y_new, indx_new

def binarize(matrix):
    copy_matrix=copy.copy(matrix)
    copy_matrix[copy_matrix > 0] = 1
    return copy_matrix

def get_train_test_dist_matched(data_file, dir_to_save):
    print "loading data..."
    data=np.loadtxt(data_file, dtype=str)
    X=data[:,:6]
    y=data[:,6]
    y=y.reshape((y.shape[0],1))
    indx=data[:,7:9]

    print "splitting data..."
    X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(X, y, indx, test_size=0.20, random_state=42)
    #X_train_new, y_train_new, indx_train_new = get_pairs_distance_matched(X_train, y_train, indx_train, 10000, 2000000, 10000, 1)    
    #X_test_new, y_test_new, indx_test_new = get_pairs_distance_matched(X_test, y_test, indx_test, 10000, 2000000, 10000, 1)    
    print "distance matching train data..."
    X_train_new, y_train_new, indx_train_new = get_pairs_distance_matched(X_train, y_train, indx_train, 110000, 120000, 10000, 1)    
    print "distance matching test data..."
    X_test_new, y_test_new, indx_test_new = get_pairs_distance_matched(X_test, y_test, indx_test, 110000, 120000, 10000, 1)    

    DATADIR=dir_to_save
    np.savetxt(DATADIR+'site1_train.bed', X_train_new[:,:3], delimiter="\t", fmt="%s") 
    np.savetxt(DATADIR+'site2_train.bed', X_train_new[:,3:], delimiter="\t", fmt="%s") 
    np.savetxt(DATADIR+'site1_test.bed', X_test_new[:,:3], delimiter="\t", fmt="%s") 
    np.savetxt(DATADIR+'site2_test.bed', X_test_new[:,3:], delimiter="\t", fmt="%s") 
    #np.savetxt(DATADIR+'labels.txt', y_new, delimiter="\t", fmt="%s") 
    np.savetxt(DATADIR+'indx_train.bed', indx_train_new, delimiter="\t", fmt="%s") 
    np.savetxt(DATADIR+'indx_test.bed', indx_test_new, delimiter="\t", fmt="%s") 

    np.save(DATADIR+'labels_train.npy', binarize(y_train_new.astype(float)).astype(int))
    np.save(DATADIR+'labels_test.npy', binarize(y_test_new.astype(float)).astype(int))

#    import pandas as pd
#    regions_train1_df = pd.DataFrame(data=X_train_new[:, :3])
#    regions_train2_df = pd.DataFrame(data=X_train_new[:, 3:])
#    regions_train1 = list(BedTool.from_dataframe(regions_train1_df))
#    regions_train2 = list(BedTool.from_dataframe(regions_train2_df))
#    
#    regions_test1_df = pd.DataFrame(data=X_test_new[:, :3])
#    regions_test2_df = pd.DataFrame(data=X_test_new[:, 3:])
#    regions_test1 = list(BedTool.from_dataframe(regions_test1_df))
#    regions_test2 = list(BedTool.from_dataframe(regions_test2_df))
#
#    y_train_bi = binarize(y_train_new.astype(float)) 
#    y_test_bi = binarize(y_test_new.astype(float)) 
#    return regions_train1, regions_test1, regions_train2, regions_test2, y_train_bi.astype(int), y_test_bi.astype(int)
#
# 
