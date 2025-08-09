import { lazy } from "react"

export const lazyLoad = (path, namedExport) => {
    return lazy(() => {
        const promise = import(path)
        if(namedExport == null) return promise.then(module => ({ default: module.default }))
        else return promise.then(module => ({ default: module[namedExport] }))
    })
}