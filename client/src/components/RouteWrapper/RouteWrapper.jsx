import { Outlet } from "react-router-dom"
import Footer from "../Footer/Footer"
import Header from "../Header/Header"
import { Suspense } from "react"
import Loader from "../Loader/Loader"

const RouteWrapper = () => {
    return (
        <>
            <Header />

            <Suspense fallback={<Loader />}>
                <Outlet />
            </Suspense>

            <Footer />
        </>
    )
}

export default RouteWrapper