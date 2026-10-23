"""SI geometry: pedestrian disks and axis-aligned rectangular vehicles.

The car stays on its initial straight axis. Interaction approximations never
replace exact disk/rectangle checks in the recorded diagnostic.
"""
import numpy as np

EPS = 1e-10


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), EPS)


def pair_geometry(p, radii, half_extents):
    """Signed surface gap and normal from j toward i, for every ordered pair."""
    delta = p[:, None, :] - p[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    normals = delta / np.maximum(distance[..., None], EPS)
    gaps = distance-radii[:, None]-radii[None, :]
    rect = np.flatnonzero(half_extents[:, 0] > 0)
    for j in rect:
        d = p-p[j]
        combined_extent = half_extents[j]+half_extents
        q = np.abs(d)-combined_extent
        nearest = np.clip(d, -combined_extent, combined_extent)
        outward = d-nearest
        outside = np.linalg.norm(outward, axis=-1)
        sdf = outside + np.minimum(np.max(q, axis=-1), 0)
        normal = outward/np.maximum(outside[:, None], EPS)
        inside = outside < EPS
        axes = np.argmax(q, axis=-1)
        for i in np.flatnonzero(inside):
            normal[i] = 0
            normal[i, axes[i]] = 1. if d[i, axes[i]] >= 0 else -1.
        gaps[:, j] = sdf-radii
        gaps[j, :] = gaps[:, j]
        normals[:, j] = normal
        normals[j, :] = -normal
    np.fill_diagonal(gaps, np.inf)
    normals[np.arange(len(p)), np.arange(len(p))] = 0
    return gaps, normals


def swept_gaps(p, next_p, radii, half_extents):
    """Exact signed-gap minimum over each linear step, including two rectangles."""
    i, j = np.triu_indices(len(p), 1)
    start, delta = p[i]-p[j], (next_p[i]-next_p[j])-(p[i]-p[j])
    fraction = np.clip(-np.sum(start*delta, axis=1)/np.maximum(np.sum(delta**2, axis=1), EPS),0,1)
    gaps = np.linalg.norm(start+fraction[:,None]*delta,axis=1)-radii[i]-radii[j]
    for k in np.flatnonzero((half_extents[i,0]>0)|(half_extents[j,0]>0)):
        rect = i[k] if half_extents[i[k],0]>0 else j[k]
        disk = j[k] if rect == i[k] else i[k]
        a = p[disk]-p[rect]
        b = (next_p[disk]-next_p[rect])-a
        extent = half_extents[rect]+half_extents[disk]
        # Distance squared to an AABB is piecewise quadratic. Split at faces.
        knots=[0.,1.]
        for axis in range(2):
            if abs(b[axis])>EPS:
                for edge in [-extent[axis],0.,extent[axis]]:
                    t=(edge-a[axis])/b[axis]
                    if 0<t<1:knots.append(float(t))
        # Interior signed-distance minimum may be where the two face distances tie.
        for sx in [-1.,1.]:
            for sy in [-1.,1.]:
                denominator=sx*b[0]-sy*b[1]
                if abs(denominator)>EPS:
                    t=(extent[0]-extent[1]-sx*a[0]+sy*a[1])/denominator
                    if 0<t<1:knots.append(float(t))
        knots=sorted(knots); candidates=list(knots)
        for lo,hi in zip(knots[:-1],knots[1:]):
            middle=a+.5*(lo+hi)*b
            outside=np.abs(middle)>extent
            if outside.any():
                offset=a[outside]-np.sign(middle[outside])*extent[outside]
                slope=b[outside]
                t=-float(offset@slope)/max(float(slope@slope),EPS)
                candidates.append(float(np.clip(t,lo,hi)))
        q=np.abs(a+np.array(candidates)[:,None]*b)-extent
        sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
        gaps[k]=float(sdf.min())-radii[disk]
    return gaps


def lateral_clearance(p, radii, extents, width):
    if width is None:return np.full(len(p),np.inf)
    halfwidth=np.where(extents[:,0]>0,extents[:,1],radii)
    return width/2-np.abs(p[:,1])-halfwidth


def ray_headways(p,heading,radii,extents,buffer=0.):
    """Distance along ego heading to another expanded footprint.

Disk/disk is an exact ray-circle intersection. Rectangle/disk uses the
axis-aligned bounding box of the rounded Minkowski sum (conservative corners).
"""
    rel=p[None,:,:]-p[:,None,:]
    along=np.sum(rel*heading[:,None,:],axis=-1)
    r=radii[:,None]+radii[None,:]+buffer
    perpendicular2=np.sum(rel*rel,axis=-1)-along**2
    intersects=(perpendicular2<r*r)&(along>0)
    result=np.where(intersects,np.maximum(0,along-np.sqrt(np.maximum(r*r-perpendicular2,0))),np.inf)
    n=len(p)
    for i,j in zip(*np.where((extents[:,None,0]>0)|(extents[None,:,0]>0))):
        if i==j:continue
        extent=extents[i]+extents[j]+radii[i]+radii[j]+buffer
        entry=-np.inf;leave=np.inf
        for axis in range(2):
            e=heading[i,axis];d=rel[i,j,axis]
            if abs(e)<EPS:
                if abs(d)>extent[axis]:entry=np.inf;leave=-np.inf;break
            else:
                a=(d-extent[axis])/e;b=(d+extent[axis])/e
                entry=max(entry,min(a,b));leave=min(leave,max(a,b))
        result[i,j]=max(0,entry) if entry<=leave and leave>0 else np.inf
    np.fill_diagonal(result,np.inf)
    return result
