import { useContext, useEffect } from "react"
import { DataContext } from "../../context/DataContext"
import { Typography } from "@mui/material"

const Dashboard = () => {
    const { crud } = useContext(DataContext)



    useEffect(() => {
        const fetching = async () => {
            const response = await crud({
                url: "/environment/?lat=42.6977&lon=23.3219",
                method: "get"
            })

            console.log(response)
        }
        fetching()
    }, [])



    return (
        <Typography variant="h1">Hi</Typography>
    )
}

export default Dashboard