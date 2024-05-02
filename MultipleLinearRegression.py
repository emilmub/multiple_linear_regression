class MultipleLinearRegression:
    def __init__(self, x_data, y_data):
        '''
        x_data: Design matrix containing n-rows (one for each observation)
                and m-columns (one for each predictor)
        '''
        intercept_ones = np.ones((x_data.shape[1],1))
        self.x_data = np.hstack((intercept_ones,x_data))
        self.y_data = y_data
        self.residual_sum = None

    def fit(self):
        inverse = np.matmul(self.x_data.T,self.x_data)
        inverse = np.linalg.inv(inverse)
        self.beta_vec = np.matmul(inverse,np.matmul(self.x_data.T,self.y_data))

    def RSS(self):
        self.residual_sum = 0
        for x,y in zip(self.x_data, self.y_data):
            residual = y
            for b, x_j in zip(x,self.beta_vec):
                residual -= b*x_j
            self.residual_sum += residual*residual
        return self.residual_sum

    def TSS(self):
        y_mean = np.mean(self.y_data)
        self.tot_sum_sq = 0
        for y in y_data:
            self.tot_sum_sq += (y - y_mean)*(y - y_mean)
        return self.tot_sum_sq