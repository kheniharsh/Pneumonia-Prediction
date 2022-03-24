import numpy as np
from PIL import Image, ImageOps
from flask import *
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

model_ct =  keras.models.load_model('prediction_ct.h5')

@app.route('/')
def index():
    
    return render_template('index.html', val='')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBYWFRgWFhYZGBgaGhwZGRgZGhkaGBwaGBgaGRgaGBgcIS4lHB4rHxoYJjgmKy8xNTU1GiQ7QDs0Py40NTEBDAwMBgYGEAYGEDEdFh0xMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMf/AABEIAM8A9AMBIgACEQEDEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAGAAIDBAUBB//EAEEQAAEDAgQDBQcCAwYFBQAAAAEAAhEDBAUhMUESUWEGInGBkRMyobHB0fBC4VJi8RQVQ4KSsgcjcnPCFyQzU6L/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8A8ZlKUkkCCUJJ3AeRQchKFI23cdlO3DnnZBUkqajRc7RaVpgjie8iiwwgNGiAescJdqVv2Nq6Qty3w2dfRaVphueQ+yC9YVXhoz2hX2XT1PbWEN5qU20bHyQdoXLt5V72pjJUqVLYH6K0+lDfeM9EDal1AEjxgrlC7JGm+pWfe0jM8cZdR9VZw6lI9+fL5lBZubkt0KpvxBytVaR8VVfSnkEED8TchTtVir3AQTlPxhFhtJQ1jVkZzCDzW+r1JLg4+pVD+9ao/W71KMLvDeixbvB+SDMGNVf43epT249W/jd6lV7jDnN2VItI1CDZb2jrD9bvUqRvaiuP1u9SsBJARt7W3H/2O9SrTe2tyP1u9UJBdQGDe3Nx/G71Kkb29uR/iO9UFFJAc/8AqHc/xpIGSQaFvhjnfstm07NudnwomwzD2wzLUD5IktrLdAD0OzWYBEeS1bfssw/q/wDxPyKNqFvsVK+z5ZeGSAWo9kGfxt/0lWWdlW/xj/Sfutd9s8aOKj46jd/gEFVnZR2zmH4FXLbs+9v6fMEFa1i2o7MnLwC3Wdxpc4DIZxl5BAO22BuPvAtHh+Qr1R1Ki3SY5Cc+pXK+LS7OWjYDMKtcV2vaRxH5IKVxjU7lo/6Am290H6PBPSQfQqtcUZ0J8iCqzKDpyJnr+yDepOe097Mc4+a1HmWh0eeyyLe5c0Q/PaVr03HhEb8kGTf3AkBW8MbLZg+inuKE5z8lbsWQEED2Tr6Kq98fp+p9FPWuDMADVZV/dPaYzJ5DRBbbdjdrvQEJ77NtUGI8Dl6IdFd5ObPgtfC3Pa6ZgbyIHxQZd7gZaYIy2KzK2AE+6D6Er0RuIsiHlp8IlKvTBbxMkjkMv6IPLK3Zeo/9B8TAHxVCt2IfuGD/ADD6L0W6u3CQGD/MSSsSvdvkjhb6H7oAer2JfsWev7LOr9kXjdnk5ehOe86GOgXTTefe4XeIz9UHl9Ts49uoMcxmoHYG7YnzC9TNi5wgN+yhfhIGuvTRB5LWw57dpVUtjVemX+FDkhXE8KI2y57oBtJWn2LpSQexYdZHgZI1Y0j0C3aDSBkSnYVD6NLiH+Gwz14AtJlAbIK9FxnMfQrSZagxGYUTKIBKuUjAjmUHKVo3TL5+qiqsYw91onWTmfJTVapZk0+J+ngm24LnSdtED2Ngcb3QB6enPos/FsRlpa0ZbT9lNirpdw7DIAfFYF1eMb3S4A8hmf2KDEubp+evll8lWdWcdZnxVq+uWkyAZ66HxWLc3n8seJKCxccUyAQeYUltiNRmRJI/mlZn94mBl4wVJSvGl2cieeYQGdhiDXth2Ttct4+qKbVrSwTmUEYUGO5DcEcwjaxb3cs0Er6QzzMclZZwhigmSVJWB4Y0yQYmKXbKY4icyco1Qzf4wHTDZ81a7QvBfw6wPJDVS6YDqzecp+hQWRizjoAB0Jn1WnaYm0DQ+soXFywk5j0KltiDoQfAlAXjEWOIEx45Z+KIcEudpkeoQDQtTkBPwKIsLYWaOzOsZICfEsOD+8Nfmh+5wt8+6t2wxDjPA7yPXqpbguYdTCDIZg7QMxJiZ+yYzDWj91tsui4cJifiRv5prKYPQ69EGS62VWpRYNY9QtK8eXZDIfmqpm2kBBiYjbggwOuSHbm1nZGz7Sd8uao4jhwLZYMwMxzH3QAFTCmzokiI0RySQE2Cf/DS/wC2z/YFuUoQvgN5/wC2oAN/wqe/8jVt0LmUGowSpqbJI6ZqnSrEaBXWvkeKCCuJKt2zQxpccsp+3moHtlw6qW/bLB4z9vJAN4xXe4nhybvH1KF7mgTOceH3RXfXbACNSNht4oSvq5kloAHqUFd7coJPjkqr6U7z6FRXNR5z4voqb3vH6yPMoLjsOyJa7rEKNtmdAZ5JtC7eMg4kxuJ+avW1/DZIBdlttodED8MqPY4HbTodl6JgVc8Of4EE2FRryRr0R7h1rwMA3KDTpmVFfv4WOIU1BugTL9ncQeTY497n5zGwGizP7E8iYGeWo80YdpyxnDIznIeSGv7XsGxqMyc0FJtm4CSAuMtXk5jLkCFaqX/D3eFsRO/3UBvyT7sDofug0LesWcxtv+FadtijpzAI9Csq3umZZ59RkFo0WNdpn1CAtwyu18RryOvkiC+bLZ/OSE8LtSBxan4hFNnWD2Fp1Gvh9wgo0z3gVZqv0KieyDHIqd4lqCq9kqKopqzw0fJUHVggZVVfjzT3VQdxPJVqiDlSwpPPETwk7DRcTDKSDH7PXANCj/22f7GoktSvOsIvA2jSz0Yz/YET4Z2gZo4kdYy9UBpRCu0NFkWV4x4HC4Fatu5BKGS4LmKP7vCNSP6KWjqVBfuDXEnIZZnw0QBV7bOzzhZhAHvfT8C3MauRB4RKFbi5J5DyQT1LZpzE+R/JVWph7SMpHiflKpvunD3T9P6Je1c7XzlBMy0DSATHmVfp2HdEc4P7LOFQ5AjLxRPh9RnCJJIO3Ln9EFezw0B7X6AEfgRxamI5TksCg8AZGc8tt9wiO0YC0Z+iDQCjvh3FK0dU26yaEHnvauzc5zXfwz8dFh0MLc4xMdeco+xMBwOnmsaS3ieAC7bL8zQB9zhh4iSQR+QqxoFugEeK1r+q8Q3cjPLQbCeapCvGXCD1zQV6Vo85kED80C2LEBmkg7nQqFl03Kcj1Vui9r8gQfQoN3DcT/Sc+o/M0XYWwRI3GqCsOtAXZZfJGOGu4IadDl+6B1y3vrrwp7lnen18QonnIlBnXIkrOqrQrOWRd1wNUFK5Krf25zf5hyP0OyqX2IiTwqg64lBujEWdR0gH4pLA4kkAJZX/AHWg7AD0ELcs68oKpvgrewu40QG+H3DmmWuI8EbYJi5JDX66B3PoUAYe+YRHZoPSbTWfL1WL2hJfp+mRC0cJq8TAdxE+irXEDiB9T80AdXLtDl1Wbc2TTnJ8dPgtLGqzGunM+Gg+6H6+IkaaIIalq0HPi+iY6xBgtkjeTmmPui+e8fIJlMRnLhyzyQTNoDij01RbhWFu4AYHPVDlP3g4HLprP0RJT/5bA6fOfyfBBYbRzyEwR80VW7O6EMYZULnNn9RnyGqLKZyAQTs8Urkd1dYu3HuoMV9qsl1BzXEbfkIgJUFe1LhM5jT90Ahf4eXHWT0GnRZT8OLdGomr27+PKZ1yyVllIcObc9f6oAatYOyJEc9kqVEjKCOp+iJLu1iDmT8lWa9rdIJ5fYILGFXRZE6cj90VYbcNqERtt9UHMoh2uSI8JpOZ3vT90BLc/UlZ19cBjC4mFoXDpaD0B9dfigjtVdFxDAchmepQZeMdo3EkM057fuhS5v3uMue49Jy9Fauwsa5fCCT+8I971UtO/HPJDN5dKjTvnNPRB6Ay5EJISpYplqkgwVcw+rBhU0+m6CCg9Dwd8gItsQgTs9WyCO8NOiA4wiG0yTpBPk391gY9dk98HTbZbGZtnAa+z/8AKSsFnCWmczvP5kgxqlXjGQnxWJc2bwTll8vFbF4/gcc/Aqq/EGkRGfx8kGQaEaENPQfurVOh/E7LyzTXvBPdBHjp6LtNh14iBtkMz9EGvhVk2eLOeX1K2rmm2ABAjOPzyWHb13BpDTPzKl9o50SIO56bICPB6bs3HoB4bomboMlhYK08IRGBkEHWHp8V2r7qawJ9U91BUDASk5g00XfaJpMII32+UzmqVSgfJaNV+WSzritA6oMu9oSVlXNkSDl4TutqpcNPMdCmBg1kHogxLGi9pG/8p+hRVZVgQG6O5fOOay61RoyAz9Ey0Y7j4thnPXZAYVxw0/BsfFAWNZvcjp7+OjPME+bUDYoM3eKAUv0MYnWhEuKOiUD4rXzKDOr1JKiSXSg4kkkgSRSSQE/ZytmF6ThLsgvJ8AqQ8DqvUsFdkEB7bwKInSJPgdPmgLFMRLHkNGQ+I2KLscueCk0c3x/paIQbitqX94fuRugq3FT2rdeqynsIM8MnY7eKs0zwHp1Vl9Vrh9voEFWiHuMGI3Oq02U2NbBPLT5LPe9zfD4qJldzjxHTYddvJAQWxY3iy7x0UzXtDWh2Zg5eaw7N/Ce8SXHflyUtJxkyc9Pqfkg9GwtwLWkclrh2SHuzzw5jTtG/REIQPpfmS7ce6VylqnXHuoKQGSXCmEpzHH1QcqtyjdZlzTkQZ6FXLioQZhUa116dUGZcvjoVn3N1Gk5/hWhdPa/oR6FZdalJzEfQoK7LxxdGvOVv2FywiAc+R1noh6pS4M/6/FQ2lVz3jUZ+fig9MtpFEHo4nzlBuJjVGNtU46HUgt8wMvgg/FTqgB8cqQCgG8fLkX9pasSgpxkygaulcSQJJdXECSSSQXcMfD/zZerYBUloXkNu6HDxXqXZarLQgPMbp8dBjuRDvVoBPqhd940gtJzCKMVqcNuG7kFo9JleWXFVzXnmJGZ9UE9+4tJzgH+qgoXHCeLp81Mx7XjmeWq42w73fyZMmNYGyCzbV2lsmYz1PyUkl8EMAA20VdsSYGUiPDqrTLwNblmZ/qgltrMySdfwrRtbcAZiZ3lZ1tVqVIDGu390R6uW3h+AVCQXwM5jU+qAl7PNlngfgiCJCysMteDJawCBUtVJcjulNYM064zaUGeBknNCqPfsuNkjdBJcty1WJcg6RK0Kj3nQGFn1qVQ6MnP8lBlVxqRkZ0Varfd3aRp16Fa1TDqhGg8EO4jh725kFsb6geJQRVb2d8uW3n1WxhNARJA/ZDDKJLu8InfY+a1rfEDThv54oD/CD3HDYPEf6TKF8ZdkUTYOR/Z+Ib974IRx2pDSg8w7T1syPJDS1ccq8T/NZSBLq4kg6uLq4gQSSCSDoK9G7I1ZaF5wjnsRVkR+ckHovau54RROx459W5/JBuIWJeeMab/dF3aKlx0qZ2br/mAHzCwTXAaWu2GnTZBl28MMgdNpKsPueLIZkjYSfROs8FqVXyZazYxmZ5ckaYL2eZT0b4k5lAI22AVqmQHA3edfRFWF9kmM94cR6ortrMDOFdZTAQZdthjW6AAK621AU5lNMoOMpwQp3NyTKZzU5QRMYpajckmhOMQgz/ZblRPcRop7l8aKtxFBA6s/nC6ys6czKe6oNwmmmNkFhrWuVavYgzI8lKzJXGZhAIYjgYglgg8tj4oUqWLuOCCHTmOnTovU61MFZlzhrX97Rw0KBYLlblvUfY/RA3aivDHI5tQWsO2cfEE/JeX9srmGn86oPOb1/E9x6wq6cTOaagSSSSDq4kkgQSSSQJE/Yyvwvjqhha3Z2pw1R5IPbqkG3BccuF0+uXxWBgGDOrP9o8Q0HutO/UreoUvaW1MbcTeLqO8SPUBEVrbhgEBBLa2TQ0CFdp0AEqQUnEg6kAugJwCDnCmlqlhdDEELG5qwWLjQnkIGQnQuQnIM66ZmqjmlalbVReylBkVJ3Ca2Z8VqvtZ1SNryGiCox/8ARW7XVMfaGZhTWtLhKDr2qvXbAV2oFSrZoKV+6KJcORPn7v1leJ9uK235nkvZcTqdx45T/tP2XhHbOtNSOqAZSSXUHEl1cQJJJJAgkkF1BxW8NdFRv51VVPov4XA8iCg997NVOKgzpHwJ+jkWWeYQL2KrTTA6ft8oRzbZNQW1KxqgYVYYUDgE9oSaE6UCAXSkEoQNhP4VwBPIQRwnBiXCnAIIXsXGMhSOXECBSIShJAxxTFK4KEoGlyr1xGamcoK5lp6IB7FqnDTeec/KP/JeCdoqvFVPn817Z2rrcNIgfn4V4PiL5qP8Y9EFZJJJAkkkkHEkkkCC6uBdQJIpJIPWv+H11xU2ic4HroV6bSfMBeL/APDu5ghvU/New2NSUGmxWGKtTKmaUFgFPCjYpWoEEiYXQkRKBvGnNcUm0wpUDEuFIlOagiLUoUjgmuQMc+E32gULik6mdkEpeE1xlZ1RxBV2kZbKCJ5VZ7tfBWapWfdOQBHbKtDM/wAjNCPZfE6jbF7adCrVmvXNV7W1HUWMrUaVPiqsa0irEPdwk5cIO62+39aGO/6XfJBvZ/tDRo21SjVY+oT7RzGj2fAHVKTaYcHkcdNwiSWnvDIjKUGz26of2gUPY2t1TewuoU2PoPBfQY0PplsNHugVCWmTBmYCB22VQhpFN5a4Oc13C6HNpgmo5pjMNAMkaRmi3D+2dOiadQU3uq07anbtDncLONtV1Sq/iaeLNsNHPidKtVu2lv7N1FlF/svZ3Yp8XCDTq3Xtg0tg+4GVnMI34WnZAH1sGuGQH29ZhIc4cVN7ZawS4iRoBmTsmf3XX4uH2NTiDhTLeB3FxkFwZETxQCY1gI0vu2lufb+zZXIruqvdxlncfUt6lFjWNBPd/wCZJdM90ZKW5/4g03moXUqgdUrViXsLWv8AYVKdWlTM6e2ptqBoOhbTAndB57dWz6biyoxzHDVrgWuE5iWnMJLU7U4nTr1mupNLGtpsZ383OLBHEQCQ2cu6DASQf//Z"
    
    predict_ct = []    
    img = Image.open(file)
    gray = ImageOps.grayscale(img)
    img = gray.resize((150,150), Image.ANTIALIAS)
    imgs = np.asarray(img)
    predict_ct.append(imgs)
    predict_ct = np.array(predict_ct)
    predict_ct = predict_ct.reshape(predict_ct.shape[0], 150, 150,1)

    predict_x=model_ct.predict(predict_ct) 
    classes_x=np.argmax(predict_x, axis=1)
    
    if classes_x[0]==0:
        render_template('home.html', val='Your Report is Normal')
    else:
        render_template('home.html', val='Oops. Report Consist Pneumonia, Consult a Doctor')

if __name__ == '__main__':
    app.run(port=8000)
