import { Typography } from "@mui/material"
import { useContext, useEffect } from "react"
import { useParams } from "react-router-dom"
import { DataContext } from "../../context/DataContext"

const Activate = () => {
    const { crud } = useContext(DataContext)
    const { uidb64, token } = useParams()

    useEffect(() => {
        const activate = async () => {
            const response = await crud({
                url: `/users/activate/${uidb64}/${token}/`,
                method: "get"
            })

            console.log(response)
        }

        activate()
    }, [])

    return (
        <Typography variant="h1">ACTIVATE</Typography>
    )
}

export default Activate