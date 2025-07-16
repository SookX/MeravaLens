import { createContext, useState } from "react";
import axios from 'axios'
import { useNavigate } from "react-router-dom";
import Loader from "../components/Loader/Loader";

export const DataContext = createContext({ })

const DataProvider = ({ children }) => {
    // Navigates users to different routes
    const navigate = useNavigate()



    // Gets the JWT tokens if the user has logged in
    const [refresh, setRefresh] = useState(localStorage.getItem('refresh') || sessionStorage.getItem('refresh') || null)
    const [access, setAccess] = useState(localStorage.getItem('access') || sessionStorage.getItem('access') || null)



    // Holds the loading state for the site
    const [loading, setLoading] = useState(false)



    return (
        <DataContext.Provider
            value={{
                navigate,
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