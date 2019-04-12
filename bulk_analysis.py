import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from astropy.stats import circcorrcoef
import astropy.units as u
import pycircstat


class gridCells:

    def __init__(self, allspikes, type, control):
        assert dir is not None

        self.allspikes = allspikes
        self.type = type
        self.control = control

        self.move_thresh = 0.01

    def mean_phase_map(self, arr, bin_size):
        
        mpm_dict = {}
        for ybin in range(0,int(self.allspikes[:,3].max()+1),bin_size):
            mpm_dict[ybin] = {}
            for xbin in range(0,int(self.allspikes[:,2].max()+1),bin_size):
                phases = []
                mpm_dict[ybin][xbin] = []
                for spike in arr:
                    if (xbin <= spike[0] <= xbin+bin_size) and (ybin <= spike[1] <= ybin+bin_size):
                        phases.append(spike[2])      
                mpm_dict[ybin][xbin] = pycircstat.mean(np.asarray(phases))

        #Rotate the dataframe 90 CCW
        mpm = pd.DataFrame.from_dict(mpm_dict).T
        phase_df = mpm.reindex(index=mpm.index[::-1])

        #(Uniform) Smoothing algorithm
        #Replaces each value with circular mean of (inclusive) neighbouring 3x3 matrix
        np_dfp = np.pad(phase_df.as_matrix(), 1, 'constant', constant_values=np.nan)
        a = np.zeros((np_dfp.shape[0], np_dfp.shape[1]))

        for i in range(len(np_dfp)-2):
            for j in range(len(np_dfp[i])-2):
                arr = np_dfp[i:i+3,j:j+3]
                if not np.isnan(arr[1,1]):
                    n_arr = arr[~np.isnan(arr)]
                    avg = pycircstat.mean(n_arr)
                    a[i+1,j+1] = avg

        a[a==0]=np.nan
        a = a[1:-1,1:-1]-3.14
        phase_df = pd.DataFrame(a)

        return phase_df

    def mean_var_map(self, arr, bin_size):
        """Bins data in a 2x2 matrix to the phase variance"""
        vm_dict = {}
        for ybin in range(0, int(arr[:, 1].max() + 1), bin_size):
            vm_dict[ybin] = {}
            for xbin in range(0, int(arr[:, 0].max() + 1), bin_size):
                phases = []
                vm_dict[ybin][xbin] = []
                for spike in arr:
                    if (xbin <= spike[0] <= xbin + bin_size) and (ybin <= spike[1] <= ybin + bin_size):
                        phases.append(spike[2])
                vm_dict[ybin][xbin] = (pycircstat.var(np.asarray(phases))+0.5)/2

        #Rotate the dataframe 90 CCW
        vm = pd.DataFrame.from_dict(vm_dict).T
        var_df = vm.reindex(index=vm.index[::-1])

        return var_df

    def adjacent_matrix(self, cell, phase):
        """Determines change vector from central cell to cell
        nearest in value in 5x5 IN FORM **[X,Y]** """        
        x = int(cell[0])
        y = int(cell[1])
        y_size = arena_size[0]-1
        a = padded_phase_df.iloc[y_size-y:y_size-y+5,x:x+5]

        try:
            nearest = np.nanargmin(np.abs(a-phase))
            loc = [(nearest%5)-2,2-(nearest//5)]

            #Rounding down to the nearest bin, adding 0.5 to point to center of bin        
            xp = x+loc[0]+0.5
            yp = y+loc[1]+0.5    
            return[xp - cell[0],yp - cell[1]]

        except:
            return [0,0]    


    def adjacent_spikes(self, spikes, phase):
        """Get location of spike with most similar phase"""
        y_size = self.arena_size[0] - 1
        phases = []
        vars = []
        for i in spikes:
            x = int(i[0])
            if x >= self.arena_size[1]:
                x -= 1

            y = int(i[1])
            if y >= self.arena_size[0]:
                y -= 1
                
            phases.append(self.phase_df.iloc[y_size - y, x])
            #vars.append(self.var_df.iloc[y_size - y, x])
        phases = np.asarray(phases)
        #vars = np.asarray(vars)
        #diffs = np.abs(phases - phase)*vars

        try:
            #nearest = np.nanargmin(diffs)
            nearest = np.nanargmin(np.abs(phases - phase))
        except:
            nearest = 0

        try:
            #Rounding down to the nearest bin, adding 0.5 to point to center of bin
            x = int(spikes[nearest][0])+0.5 
            y = int(spikes[nearest][1])+0.5
            return [x - spikes[0, 0], y - spikes[0, 1]]

        except:
            return [0.0, 0.0]
    
    

    @staticmethod
    def vector_angles(df):
        """Basic formula for determining angle between vectors"""

        angles = []
        for index, row in df.iterrows():
            p0 = [row['Xdif'], row['Ydif']]
            p1 = [0, 0]
            p2 = [row['Xdif Predicted'], row['Ydif Predicted']]
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            atan = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            angles.append(np.degrees(atan))

        return angles

    @staticmethod
    def abs_vector_angles(arr):
        """Determines the angle between the horizontal axis (+1 x, +0 y)
        and the current vector, returns [observed,predicted]"""

        obs_angles = []
        pred_angles = []

        for i in range(len(arr) - 1):
            p0 = [arr[i, 1] + 1, arr[i, 2]]
            p1 = [arr[i, 1], arr[i, 2]]
            p2 = [arr[i + 1, 1], arr[i + 1, 2]]
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            atan = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            if np.degrees(atan) < 0:
                obs_angles.append(360 + np.degrees(atan))
            else:
                obs_angles.append(np.degrees(atan))

        for i in range(len(arr) - 1):
            p0 = [arr[i, 1] + 1, arr[i, 2]]
            p1 = [arr[i, 1], arr[i, 2]]
            p2 = [arr[i, 1] + arr[i, 7], arr[i, 2] + arr[i, 8]]
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            atan = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            if np.degrees(atan) < 0:
                pred_angles.append(360 + np.degrees(atan))
            else:
                pred_angles.append(np.degrees(atan))

        return (list(zip(obs_angles, pred_angles)))

    def phase_analysis(self):
        """Loads and normalizes data, generates predictions based on
        mean phase map and saves everything in a dataframe"""
        
        #Separate all spike data into variables
        #Splitting all spike-rows to divide the set 50% for the analysis and 50% for the mean phase map generation
        evens = self.allspikes[::2, :]
        odds = self.allspikes[1::2, :]
        
        self.spkT = evens[:, 0]
        self.phase = evens[:, 1]
        self.XYspkT = evens[:, 2:]
        
        # Normalizes position data to [0,0] alignment in bottom-left corner
        self.XYspkT[:, 1] -= self.XYspkT[:, 1].min()
        self.XYspkT[:, 0] -= self.XYspkT[:, 0].min()
        self.scaled_XY = self.XYspkT / 2
        
        self.diff = int(self.XYspkT.shape[0]/200)
        if self.diff < 6:
            self.diff = 6

        # Load precise trajectory
        #self.xyPos[:, 1] -= self.xyPos[:, 1].min()
        #self.xyPos[:, 0] -= self.xyPos[:, 0].min()

        # Load phase data
        if self.control == True:
            self.phase = np.random.rand(self.spkT.size)*6.28
        
        self.scaled_phase = self.phase - 3.14
        
        # Generate mean phase map
        #vm_arr = np.column_stack((self.XYspkT, self.phase))
        #self.var_df = self.mean_var_map(vm_arr, 2)
            
        # Generate mean phase map
        XY_odds = odds[:,2:]
        XY_odds[:,1] -= XY_odds[:,1].min()
        XY_odds[:,0] -= XY_odds[:,0].min()
        mpm_arr = np.column_stack((XY_odds, odds[:,1]))
        self.phase_df = self.mean_phase_map(mpm_arr, 2)
        self.arena_size = self.phase_df.shape

        # Main analysis!
        # 1) Combine all data and sort by spike times
        unsorted = np.column_stack((self.spkT, self.scaled_XY, self.scaled_phase))
        sorted = unsorted[unsorted[:, 0].argsort()]

        # 2) Calculate movement magnitudes
        xdif = np.append(sorted[1:, 1], 0) - np.append(sorted[:-1, 1], 0)
        ydif = np.append(sorted[1:, 2], 0) - np.append(sorted[:-1, 2], 0)

        # 3) Drop rows with movements below threshold
        raw = np.column_stack((sorted, xdif, ydif))
        movement = raw[np.any(abs(raw[:, 4:]) >= self.move_thresh, axis=1)]

        # 4) Recalculate movement magnitudes as in 2)
        xdif = np.append(movement[1:, 1], 0) - np.append(movement[:-1, 1], 0)
        ydif = np.append(movement[1:, 2], 0) - np.append(movement[:-1, 2], 0)

        # 5) Combine all data, load next spikes phases in row for convenience
        next_phase = np.insert(movement[1:, 3], -1, 0)
        combined = np.column_stack((movement[:, :4], next_phase, xdif, ydif))

        # 6) Generate predictions

        # Spatial Analysis
        if self.type == 'spatial':
            predicted = [self.adjacent_matrix([i[1], i[2]], i[4]) for i in combined]
            predicted_movement = np.asarray(predicted)

        # Temporal Analysis
        elif self.type == 'temporal':
            predicted = [self.adjacent_spikes(combined[i:i + self.diff, 1:3], combined[i, 4]) for i in
                         range(len(combined))]
            predicted_movement = np.asarray(predicted)

        # 7) Load all data into dataframe
        self.all = np.column_stack((combined, predicted_movement))
        self.df = pd.DataFrame(data=self.all,
                               columns=['Time', 'X', 'Y', 'Phase', 'Next Phase', 'Xdif', 'Ydif', 'Xdif Predicted',
                                        'Ydif Predicted'])

        self.angles = np.asarray(self.abs_vector_angles(self.all))

        # Generate observed/predicted circular correlation coefficient

        self.rl, p = pearsonr(self.angles[:, 0], self.angles[:, 1])
        self.rc = pycircstat.corrcc(np.radians(self.angles[:, 0]), np.radians(self.angles[:, 1]))

        
        self.heatmap, xedges, yedges = np.histogram2d(self.angles[:, 0], self.angles[:, 1], bins=30)


class figureGenerator:

    def __init__(self, angles):

        self.angles = angles


    def corr_plot(self):
        """Generates simple bivariate distribution for correlation"""
        corr_df = pd.DataFrame(data=self.angles, columns=['Observed Heading Direction', 'Predicted Heading Direction'])
        sns.jointplot(x='Observed Heading Direction', y='Predicted Heading Direction', data=corr_df, kind='kde')
        plt.ylim(0, None)
        plt.xlim(0.1, None)
        plt.show()

    def corr_hex(self):
        """Generates simple bivariate distribution for correlation (hex form)"""
        corr_df = pd.DataFrame(data=self.angles, columns=['Observed Heading Direction', 'Predicted Heading Direction'])
        sns.jointplot(x='Observed Heading Direction', y='Predicted Heading Direction', data=corr_df, kind='hex')
        plt.show()

    def corr_heatmap(self):
        """Generates correlation heatmap"""
        heatmap, xedges, yedges = np.histogram2d(self.angles[:, 0], self.angles[:, 1], bins=30)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='afmhot')
        plt.colorbar()
        plt.ylabel('Predicted Heading Direction')
        plt.xlabel('Observed Heading Direction')
        plt.show()