 
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(ious[0].reshape(200, 176, 2)[:,:,0], interpolation='none')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(ious[0].reshape(200, 176, 2)[:,:,1], interpolation='none')
        plt.colorbar()
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(ious[1].reshape(200, 176, 2)[:,:,0], interpolation='none')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(ious[1].reshape(200, 176, 2)[:,:,1], interpolation='none')
        plt.colorbar()

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(pos_equal_one[:,:,0], interpolation='none')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(neg_equal_one[:,:,0], interpolation='none')
        plt.colorbar()

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(pos_equal_one[:,:,1], interpolation='none')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(neg_equal_one[:,:,1], interpolation='none')
        plt.colorbar()