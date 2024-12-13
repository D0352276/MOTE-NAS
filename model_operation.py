import numpy as np

def Training(model,train_data,batch_size=32,validation_data=None,epochs=1,step_per_epoch=1,verbose="auto",callbacks=[]):
    def gen():yield 1
    if(type(train_data)==type(gen())):
        model.fit(train_data,
                  validation_data=validation_data,
                  epochs=epochs,
                  steps_per_epoch=step_per_epoch,
                  verbose=verbose,
                  max_queue_size=32,
                  workers=1,
                  shuffle=False,
                  use_multiprocessing=False,
                  callbacks=callbacks)
    elif(type(train_data)==list or type(train_data)==tuple):
        model.fit(*train_data,
                  validation_data=validation_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  callbacks=callbacks)
        
def CifarPredict(model,labels,img):
    code=model.predict(np.array([img]))[0]
    label_idx=np.argmax(code)
    return labels[label_idx]