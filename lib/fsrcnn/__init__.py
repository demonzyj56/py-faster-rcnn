from .fsrcnn import SrNetHolder

__net_holder = SrNetHolder()
FSRCNN = lambda im, scale: __net_holder.im_upscale(im, scale)
__all__ = ['FSRCNN']
