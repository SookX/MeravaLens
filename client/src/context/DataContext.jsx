import { createContext, useState } from "react";
import Loader from "../components/Loader/Loader";

export const DataContext = createContext({ })

const DataProvider = ({ children }) => {
    // Gets the JWT tokens if the user has logged in
    const [refresh, setRefresh] = useState(localStorage.getItem('refresh') || sessionStorage.getItem('refresh') || null)
    const [access, setAccess] = useState(localStorage.getItem('access') || sessionStorage.getItem('access') || null)



    // Holds the loading state for the site
    const [loading, setLoading] = useState(false)



    return (
        <DataContext.Provider
            value={{
                access, setAccess, setRefresh,
                setLoading
            }}
        >
            { loading && <Loader /> }
            {children}
        </DataContext.Provider>
    )
}

export default DataProvider